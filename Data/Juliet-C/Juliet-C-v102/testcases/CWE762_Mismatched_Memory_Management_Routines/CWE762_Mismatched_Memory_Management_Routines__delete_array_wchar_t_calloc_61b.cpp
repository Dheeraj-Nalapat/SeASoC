/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE762_Mismatched_Memory_Management_Routines__delete_array_wchar_t_calloc_61b.cpp
Label Definition File: CWE762_Mismatched_Memory_Management_Routines__delete_array.label.xml
Template File: sources-sinks-61b.tmpl.cpp
*/
/*
 * @description
 * CWE: 762 Mismatched Memory Management Routines
 * BadSource: calloc Allocate data using calloc()
 * GoodSource: Allocate data using new []
 * Sinks:
 *    GoodSink: Deallocate data using free()
 *    BadSink : Deallocate data using delete []
 * Flow Variant: 61 Data flow: data returned from one function to another in different source files
 *
 * */

#include "std_testcase.h"

namespace CWE762_Mismatched_Memory_Management_Routines__delete_array_wchar_t_calloc_61
{

#ifndef OMITBAD

wchar_t * bad_source(wchar_t * data)
{
    /* POTENTIAL FLAW: Allocate memory with a function that requires free() to free the memory */
    data = (wchar_t *)calloc(100, sizeof(wchar_t));
    return data;
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B() uses the GoodSource with the BadSink */
wchar_t * goodG2B_source(wchar_t * data)
{
    /* FIX: Allocate memory using new [] */
    data = new wchar_t[100];
    return data;
}

/* goodB2G() uses the BadSource with the GoodSink */
wchar_t * goodB2G_source(wchar_t * data)
{
    /* POTENTIAL FLAW: Allocate memory with a function that requires free() to free the memory */
    data = (wchar_t *)calloc(100, sizeof(wchar_t));
    return data;
}

#endif /* OMITGOOD */

} // close namespace
