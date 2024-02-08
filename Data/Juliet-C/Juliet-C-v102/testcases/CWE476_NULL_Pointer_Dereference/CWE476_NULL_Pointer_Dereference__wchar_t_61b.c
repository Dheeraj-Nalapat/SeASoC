/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE476_NULL_Pointer_Dereference__wchar_t_61b.c
Label Definition File: CWE476_NULL_Pointer_Dereference.label.xml
Template File: sources-sink-61b.tmpl.c
*/
/*
 * @description
 * CWE: 476 NULL Pointer Dereference
 * BadSource:  Set data to NULL
 * GoodSource: Initialize data
 * Sinks:
 *    BadSink : Print data
 * Flow Variant: 61 Data flow: data returned from one function to another in different source files
 *
 * */

#include "std_testcase.h"

#include <wchar.h>

#ifndef OMITBAD

wchar_t * CWE476_NULL_Pointer_Dereference__wchar_t_61b_bad_source(wchar_t * data)
{
    /* FLAW: Set data to NULL - it will be used in the sink */
    data = NULL;
    return data;
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B() uses the GoodSource with the BadSink */
wchar_t * CWE476_NULL_Pointer_Dereference__wchar_t_61b_goodG2B_source(wchar_t * data)
{
    /* FIX: Initialize data - it will be used in the sink */
    data = L"Good";
    return data;
}

#endif /* OMITGOOD */
