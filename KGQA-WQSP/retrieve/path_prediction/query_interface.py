# -*- coding: utf-8 -*-
# !/usr/bin/python

import time
import json
import re
from datetime import datetime
import requests
from SPARQLWrapper import SPARQLWrapper, JSON

kb_endpoint = "http://10.201.190.172:8890//sparql"

#@timeout(15)
def KB_query_with_timeout(_query, kb_endpoint):
    """
    :param _query: sparql query statement
    :return:
    """
    sparql = SPARQLWrapper(kb_endpoint)
    sparql.setQuery(_query)
    sparql.setReturnFormat(JSON)
    # sparql.setTimeout(5)
    response = sparql.query().convert()
    results = parse_query_results(response)
    return results
    
def KB_query(_query, kb_endpoint=kb_endpoint,max_retries=1000):
    """
    :param _query: sparql query statement
    :return:
    """
    retries=0
    while retries < max_retries:
        try:
            sparql = SPARQLWrapper(kb_endpoint)
            sparql.setQuery(_query)
            sparql.setReturnFormat(JSON)
            sparql.setTimeout(30)
            response = sparql.query().convert()
            results = parse_query_results(response)
            return results
        except Exception as e:
            #print("Query Failure: {}".format(e))
            #print('Current Query: {}'.format(_query))
            retries += 1
            
            if "timed out" in str(e):  # Check if the exception message contains "timed out"
                #print('Query timed out, please check the query carefully')
                return "timed out"  # Return None on timeout
            time.sleep(1)  # Wait for a moment before retrying
            
    return None  # Return None or raise an exc

def query_ent_name(x,kb_endpoint=kb_endpoint):
    query = "PREFIX ns: <http://rdf.freebase.com/ns/> SELECT ?name WHERE {  ns:" + x + " ns:type.object.name ?name .}"
    results = KB_query(query, kb_endpoint)
    if len(results) == 0:
        query = "PREFIX ns: <http://rdf.freebase.com/ns/> SELECT ?name WHERE {ns:" + x + " ns:common.topic.alias ?name .}"
        results = KB_query(query, kb_endpoint)
        if len(results) == 0:
            return None
    name = results[0]["name"]
    return name

def parse_query_results(response):

    #print(response)
    if "boolean" in response:  # ASK
        results = [response["boolean"]]
    else:
        if len(response["results"]["bindings"]) > 0 and "callret-0" in response["results"]["bindings"][0]: # COUNT
            results = [int(response['results']['bindings'][0]['callret-0']['value'])]
        else:
            results = []
            for res in response['results']['bindings']:
                res = {k: v["value"] for k, v in res.items()}
                results.append(res)
    return results

