Clone from [edeca/querydict](https://github.com/edeca/querydict) repository.

# querydict

[![Build Status](https://travis-ci.org/edeca/querydict.svg?branch=master)](https://travis-ci.org/edeca/querydict)
[![Docs Status](https://readthedocs.org/projects/querydict/badge/?version=latest&style=flat)](https://querydict.readthedocs.io)

Easily match data in Python dictionaries against Lucene queries. See the [full documentation](https://querydict.readthedocs.io) on Read the Docs.

This library takes a Lucene syntax query and matches against Python dictionaries, returning `True` or `False`.
It is designed to allow selection of records that match user queries, for example in a streaming system where
data can be represented as a dictionary.
 
A simple example:

    from querydict.parser import QueryEngine
     
    john_1 = { "name": "John", "eye_colour": "Blue" }
    john_2 = { "name": "John", "eye_colour": "Green" }
    
    q = QueryEngine("name:Bob AND eye_colour:Blue")
    q.match(john_1)    # => True
    q.match(john_2)    # => False

More complicated queries are possible, including nested dictionaries:

    data = { "foo": { "bar": { "baz": { "wibble": "wobble" }}}}
    q = QueryEngine("foo.bar.baz.wibble:wobble")
    q.match(data)    # => True

And grouping inside the query:

    england = { "country": "England", "continent": "Europe", "weather": "Rainy" }
    spain = { "country": "Spain", "continent": "Europe", "weather": "Sunny" }
    
    q = QueryEngine("(continent:Europe AND weather:Sunny) OR country:England")
    q.match(england)    # => True
    q.match(spain)      # => True

# Query syntax

Please see the [Lucene documentation](https://lucene.apache.org/core/2_9_4/queryparsersyntax.html) for details of the 
query syntax. Note this module approximates Lucene queries, and isn't designed to replicate exactly how Lucene works.

If more powerful queries are required it would be worth investigating [jsonpath](https://jsonpath.com/), or ingesting 
data into Elasticsearch and using Kibana. 

## Differences from Lucene

This module has the following differences from Lucene queries: 

* Wildcard searches using `?` and `*` are not currently supported. This will be added for v1.0.
* Fuzzy searches using `~` are not currently supported. This will be added for v1.0.
* Range searches using `[..]` or `{..}` are not currently supported. This will be added for v1.0.
* Boosted terms using `^` are not supported. Because the module does not score documents, these are silently ignored.
* Field grouping is not supported. Support will be considered.
* Proximity searches using `~` are not supported. Support will be considered.

# Data format

The following data formats are well supported inside the dictionary:

* Strings.
* Integers (by v1.0).
* Datetime objects (by v1.0).
* Nested dictionaries.

Lists (arrays) are not well supported, as there is no compatible Lucene query syntax. Queries can match a specified
object in a list, for example:

    data = { "list": [ { "name": "cat" }, { "name": "dog" } ] }
    q = QueryEngine("list.1.item:dog")
    q.match(data)    # => True

However, there is no way to match any item in a list. 

Kibana query syntax *does* support items nested in lists (see [the documentation](https://www.elastic.co/guide/en/kibana/current/kuery-query.html#_match_a_single_nested_document))
but this is currently outside the scope of this module.

# Installation

    pip install querydict
    
Dependencies are automatically installed. Parsing of the Lucene query is handled by 
[luqum](https://github.com/jurismarches/luqum). Easy access to dictionary keys is
provided by [dotty-dict](https://pypi.org/project/dotty-dict/).

# Todo

* Implement support for different data types, e.g. integers and dates.
* Consider how best to handle nested lists (arrays).
* Implement range and fuzzy matching.
* Implement regular expression support (similar to Elasticsearch queries).
* Implement optional tokenisation for data fields, splitting up string data into multiple parts.