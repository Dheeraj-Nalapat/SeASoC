/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE590_Free_Of_Invalid_Pointer_Not_On_The_Heap__free_int_alloca_54d.c
Label Definition File: CWE590_Free_Of_Invalid_Pointer_Not_On_The_Heap__free.label.xml
Template File: sources-sink-54d.tmpl.c
*/
/*
 * @description
 * CWE: 590 Free of Invalid Pointer Not on the Heap
 * BadSource: alloca Data buffer is allocated on the stack with alloca()
 * GoodSource: Allocate memory on the heap
 * Sink:
 *    BadSink : Print then free data
 * Flow Variant: 54 Data flow: data passed as an argument from one function through three others to a fifth; all five functions are in different source files
 *
 * */

#include "std_testcase.h"

#include <wchar.h>

/* all the sinks are the same, we just want to know where the hit originated if a tool flags one */

#ifndef OMITBAD

/* bad function declaration */
void CWE590_Free_Of_Invalid_Pointer_Not_On_The_Heap__free_int_alloca_54e_bad_sink(int * data);

void CWE590_Free_Of_Invalid_Pointer_Not_On_The_Heap__free_int_alloca_54d_bad_sink(int * data)
{
    CWE590_Free_Of_Invalid_Pointer_Not_On_The_Heap__free_int_alloca_54e_bad_sink(data);
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* good function declaration */
void CWE590_Free_Of_Invalid_Pointer_Not_On_The_Heap__free_int_alloca_54e_goodG2B_sink(int * data);

/* goodG2B uses the GoodSource with the BadSink */
void CWE590_Free_Of_Invalid_Pointer_Not_On_The_Heap__free_int_alloca_54d_goodG2B_sink(int * data)
{
    CWE590_Free_Of_Invalid_Pointer_Not_On_The_Heap__free_int_alloca_54e_goodG2B_sink(data);
}

#endif /* OMITGOOD */
