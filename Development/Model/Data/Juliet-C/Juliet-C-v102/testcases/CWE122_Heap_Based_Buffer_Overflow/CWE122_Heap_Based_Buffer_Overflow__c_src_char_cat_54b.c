/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE122_Heap_Based_Buffer_Overflow__c_src_char_cat_54b.c
Label Definition File: CWE122_Heap_Based_Buffer_Overflow__c_src.label.xml
Template File: sources-sink-54b.tmpl.c
*/
/*
 * @description
 * CWE: 122 Heap Based Buffer Overflow
 * BadSource:  Initialize data as a large string
 * GoodSource: Initialize data as a small string
 * Sink: cat
 *    BadSink : Copy data to string using strcat
 * Flow Variant: 54 Data flow: data passed as an argument from one function through three others to a fifth; all five functions are in different source files
 *
 * */

#include "std_testcase.h"

#include <wchar.h>

/* all the sinks are the same, we just want to know where the hit originated if a tool flags one */

#ifndef OMITBAD

/* bad function declaration */
void CWE122_Heap_Based_Buffer_Overflow__c_src_char_cat_54c_bad_sink(char * data);

void CWE122_Heap_Based_Buffer_Overflow__c_src_char_cat_54b_bad_sink(char * data)
{
    CWE122_Heap_Based_Buffer_Overflow__c_src_char_cat_54c_bad_sink(data);
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* good function declaration */
void CWE122_Heap_Based_Buffer_Overflow__c_src_char_cat_54c_goodG2B_sink(char * data);

/* goodG2B uses the GoodSource with the BadSink */
void CWE122_Heap_Based_Buffer_Overflow__c_src_char_cat_54b_goodG2B_sink(char * data)
{
    CWE122_Heap_Based_Buffer_Overflow__c_src_char_cat_54c_goodG2B_sink(data);
}

#endif /* OMITGOOD */
