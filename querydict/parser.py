"""
This module implements the core of querydict, in the `QueryEngine` class. Sample usage:

    >>> from querydict.parser import QueryEngine
    >>> data = { "name", "Bob" }
    >>> query = QueryEngine("name:Bob")
    >>> query.match("data")  # True
"""

from typing import Union, NoReturn
from dotty_dict import dotty
from luqum.parser import parser, ParseSyntaxError
from luqum.tree import (
    Item,
    AndOperation,
    OrOperation,
    Group,
    SearchField,
    Word,
    Phrase,
    Fuzzy,
    UnknownOperation,
    Range,
    Not,
)
from luqum.utils import UnknownOperationResolver


class QueryException(Exception):
    """
    Exception raised when parsing a query string fails, for example if the query is invalid or uses unsupported
    features.
    """

    pass


class MatchException(Exception):
    """
    Exception raised when matching fails, for example if input dictionary contains keys with the wrong data type
    for the specified query.
    """

    pass


class QueryEngine:
    """
    Match a Lucene style query against dict data, using an abstract tree parser.

    Args:
        query: A Lucene style query
        short_circuit: Whether to terminate matching early inside AND or OR conditions (default: True)
        ambiguous_action: The action to use for ambiguous queries, for example "field1:value1 field2:value2" (default: "AND")
        allow_bare_field: Whether to allow a search term without a specified field, for example "value1" (default: False).
        max_depth: The maximum recursion depth when parsing a query (default: 10).

    Raises:
        QueryException: If the input `query` is too complex, or uses unsupported features.
    """

    supported_ops = [AndOperation, OrOperation, Group, SearchField, Word, Phrase, Not]
    ambiguous_actions = {"Exception": None, "AND": AndOperation, "OR": OrOperation}

    def __init__(
        self,
        query: str,
        short_circuit: bool = True,
        ambiguous_action: str = "AND",
        allow_bare_field: bool = False,
        max_depth: int = 10,
    ):
        """

        """

        if ambiguous_action not in self.ambiguous_actions:
            raise ValueError(
                "Invalid ambiguous_action, expected one of AND, OR, Exception"
            )

        self.short_circuit = short_circuit
        self.allow_bare_field = allow_bare_field
        self.max_depth = max_depth
        self._contains_bare_field = False
        self._parse_query(query, ambiguous_action)

    def _parse_query(self, query: str, ambiguous_action: bool) -> None:
        """
        Parse the query, replace any ambiguous (unknown) parts with the correct operation
        and check for unsupported operations.

        Args:
            query: A Lucene style query.
            ambiguous_action: The action to use for ambiguous queries, for example "field1:value1 field2:value2" (default: "AND")
        """
        if query is None or query.strip() == "":
            raise ValueError("Need a valid query")

        try:
            self._tree = parser.parse(query)
        except ParseSyntaxError as exc:
            raise QueryException(
                "Could not parse the query, error: {}".format(str(exc))
            )

        # Replace any UnknownOperation with the chosen action
        if ambiguous_action != "Exception":
            operation = self.ambiguous_actions[ambiguous_action]
            resolver = UnknownOperationResolver(resolve_to=operation)
            self._tree = resolver(self._tree)

        # Raise a QueryException if the user has passed unsupported search terms
        self._check_tree(self._tree)

    def _check_tree(self, root: Item, parent: Item = None, depth: int = 0) -> None:
        """ Check for unsupported object types in the tree

        Args:
            root: The root tree item to check.
            parent: The parent of the current root item, used to ensure certain fields are always contained in others.
            depth: Current recursion depth, used to avoid infinite loops or malicious queries.

        Raises:
            QueryException: If a query is invalid, by being too complicated or containing unsupported features.
        """

        depth += 1
        if depth > self.max_depth:
            raise QueryException(
                "Query too complicated, increase max_depth if required"
            )

        # Check for Word/Phrase objects that don't have SearchField as a parent. If configured to
        # allow this, flag at the object level so match() can require a default field.
        if any(isinstance(root, t) for t in [Word, Phrase]) and not isinstance(
            parent, SearchField
        ):
            if not self.allow_bare_field:
                raise QueryException("Query contains search term without a named field")

            self._contains_bare_field = True

        elif isinstance(root, UnknownOperation):
            raise QueryException(
                "Query contains an ambiguous (unknown) operation, use AND or OR"
            )

        elif isinstance(root, Fuzzy):
            # TODO: Implement with levenshtein matching
            raise QueryException("Fuzzy matching with ~ is not currently supported")

        elif isinstance(root, Range):
            # TODO: Implement, noting difference between [] (inclusive) and {} (exclusive)
            raise QueryException(
                "Range matching with [..] or {..} is not currently supported"
            )

        elif not any(
            isinstance(root, t) for t in self.supported_ops
        ):  # pragma: no cover
            raise QueryException(
                "Unsupported operation type {}, please file a bug".format(str(root))
            )

        for child in root.children:
            self._check_tree(child, root, depth)

    def match(self, data: dict, default_field: str = None) -> bool:
        """ Match a dictionary against the configured query.

        Args:
            data: A dictionary containing fields and values to match against.
            default_field: The name of a field to use for unqualified values.

        Returns:
            True if there is a match, False otherwise

        Raises:
            MatchException: If there is a problem with the input data dictionary.
        """
        # The user can't pass a query like "field:foo and bar" without specifying which
        # field to search for "bar".
        if self._contains_bare_field and default_field is None:
            raise MatchException(
                "Need a default_field to use for matching unqualified field"
            )

        data = dotty(data)
        return self._match(data, self._tree)

    def _match(self, data: dict, operation: Item) -> bool:
        """ Internal method that recurses through the query AST to conduct matching.

        Args:
            data: A dictionary containing fields and values to match against.
            operation: The current luqum.tree object which is being matched.

        Returns:
            True if the match succeeeds, False otherwise.
        """

        # This dictionary maps supported luqum classes to methods on this class
        op_map = {
            SearchField: self._search_field,
            AndOperation: self._and,
            OrOperation: self._or,
            Group: self._group,
            Word: self._bare_field,
            Phrase: self._bare_field,
            Not: self._not,
        }

        handler_fn = op_map.get(type(operation), None)

        if handler_fn is None:  # pragma: no cover
            raise Exception("Unhandled operation type {}".format(str(type(operation))))

        result = handler_fn(data, operation)
        return result

    @staticmethod
    def _check_children(operation: Item) -> None:
        """ Check that a given luqum.tree object only has one child.

        This is an internal sanity check to ensure that assumptions made in this module are correct.

        Args:
            operation: The luqum.tree object to check.
        """
        if len(operation.children) > 1:  # pragma: no cover
            raise Exception(
                "Unhandled operation with more than 1 child, please report a bug"
            )

    def _group(self, data: dict, operation: Group) -> bool:
        """ Match a Group object, by stripping it and calling _match() with the contents of the group.

        Args:
            data: A dictionary containing fields and values to match against.
            operation: Instance of luqum.tree.Group describing the grouped operations.

        Returns:
            True if the group matches, False otherwise.
        """
        self._check_children(operation)
        return self._match(data, operation.children[0])

    def _not(self, data: dict, operation: Not) -> bool:
        """ Invert a match to support the "NOT" query syntax.

        Args:
            data: A dictionary containing fields and values to match against.
            operation: Instance of luqum.tree.Not describing the operation to invert.

        Returns:
            The inverted result of _match().
        """
        self._check_children(operation)
        return not self._match(data, operation.children[0])

    def _and(self, data: dict, operation: AndOperation) -> bool:
        """ Implement the AND operator.

        This method will short circuit and return False instantly if any component fails to match.

        Args:
            data: A dictionary containing fields and values to match against.
            operation: Instance of luqum.tree.AndOperation describing the operations to AND together.

        Returns:
            True if the query matches, False otherwise.
        """
        result = True

        for child in operation.children:
            if not self._match(data, child):
                # Failure of one AND component means none can match, return immediately
                if self.short_circuit:
                    return False

                result = False

        return result

    def _or(self, data: dict, operation: OrOperation) -> bool:
        """ Implement the OR operator.

        This method will short circuit and return True instantly if any component matches.

        Args:
            data: A dictionary containing fields and values to match against.
            operation: Instance of luqum.tree.OrOperation describing the operations to OR together.

        Returns:
            True if the query matches, False otherwise.
        """
        result = False

        for child in operation.children:
            if self._match(data, child):
                # Success of one OR component means it's a match
                if self.short_circuit:
                    return True

                result = True

        return result

    def _bare_field(self, data: dict, operation: Union[Word, Phrase]) -> NoReturn:
        """ Match a Word or Phrase against the default field.

        Args:
            data: A dictionary containing fields and values to match against.
            operation: Instance of luqum.tree.Word or luqum.tree.Phrase.

        Raises:
            NotImplementedError: This method needs to be implemented.
        """
        raise NotImplementedError("Unimplemented")

    def _search_field(self, data: dict, operation: SearchField) -> bool:
        """
        Process a single SearchField object, which should have either a Word() or Phrase()
        as a child.  Fuzzy() and Range() are not currently supported.

        Args:
            data: A dictionary containing fields and values to match against.
            operation: Instance of luqum.tree.SearchField which contains a child object to match.

        Returns:
            True if the query matches, False otherwise.
        """

        self._check_children(operation)
        # logging.debug("Checking field %s against condition %s", op.name, op.expr)

        try:
            field = data[operation.name]
        except KeyError:
            # logging.debug("Field %s was not found in the input data", op.name)
            return False

        match = operation.children[0]

        # TODO: Check data type, e.g. for an int field we expect integer in the search query

        if isinstance(match, Word):
            return field == match.value

        if isinstance(match, Phrase):
            # Strip the quotes, first check these are always here
            if match.value[0] != '"' or match.value[-1] != '"':  # pragma: no cover
                raise Exception(
                    "Expected Phrase to start and end with double quotes, please report a bug"
                )

            return match.value[1:-1] in field

        # This should not be possible due to previous checks, but added to ensure
        # matching cannot silently fail.
        raise Exception("Unhandled SearchField child")  # pragma: no cover
