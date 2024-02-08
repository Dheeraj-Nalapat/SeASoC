/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE194_Unexpected_Sign_Extension__rand_strncpy_54b.c
Label Definition File: CWE194_Unexpected_Sign_Extension.label.xml
Template File: sources-sink-54b.tmpl.c
*/
/*
 * @description
 * CWE: 194 Unexpected Sign Extension
 * BadSource: rand Set data to result of rand(), which could be negative
 * GoodSource: Positive integer
 * Sink: strncpy
 *    BadSink : Copy strings using strncpy() with the length of data
 * Flow Variant: 54 Data flow: data passed as an argument from one function through three others to a fifth; all five functions are in different source files
 *
 * */

#include "std_testcase.h"

/* all the sinks are the same, we just want to know where the hit originated if a tool flags one */

#ifndef OMITBAD

/* bad function declaration */
void CWE194_Unexpected_Sign_Extension__rand_strncpy_54c_bad_sink(short data);

void CWE194_Unexpected_Sign_Extension__rand_strncpy_54b_bad_sink(short data)
{
    CWE194_Unexpected_Sign_Extension__rand_strncpy_54c_bad_sink(data);
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* good function declaration */
void CWE194_Unexpected_Sign_Extension__rand_strncpy_54c_goodG2B_sink(short data);

/* goodG2B uses the GoodSource with the BadSink */
void CWE194_Unexpected_Sign_Extension__rand_strncpy_54b_goodG2B_sink(short data)
{
    CWE194_Unexpected_Sign_Extension__rand_strncpy_54c_goodG2B_sink(data);
}

#endif /* OMITGOOD */
