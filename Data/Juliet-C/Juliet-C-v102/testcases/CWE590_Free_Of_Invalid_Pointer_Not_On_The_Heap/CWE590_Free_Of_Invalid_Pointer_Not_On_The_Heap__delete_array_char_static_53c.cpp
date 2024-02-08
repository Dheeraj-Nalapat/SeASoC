/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE590_Free_Of_Invalid_Pointer_Not_On_The_Heap__delete_array_char_static_53c.cpp
Label Definition File: CWE590_Free_Of_Invalid_Pointer_Not_On_The_Heap__delete_array.label.xml
Template File: sources-sink-53c.tmpl.cpp
*/
/*
 * @description
 * CWE: 590 Free of Invalid Pointer Not on the Heap
 * BadSource: static Data buffer is declared static on the stack
 * GoodSource: Allocate memory on the heap
 * Sink:
 *    BadSink : Print then free data
 * Flow Variant: 53 Data flow: data passed as an argument from one function through two others to a fourth; all four functions are in different source files
 *
 * */

#include "std_testcase.h"

#include <wchar.h>

namespace CWE590_Free_Of_Invalid_Pointer_Not_On_The_Heap__delete_array_char_static_53
{

/* all the sinks are the same, we just want to know where the hit originated if a tool flags one */

#ifndef OMITBAD

/* bad function declaration */
void bad_sink_d(char * data);

void bad_sink_c(char * data)
{
    bad_sink_d(data);
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B uses the GoodSource with the BadSink */
void goodG2B_sink_d(char * data);

void goodG2B_sink_c(char * data)
{
    goodG2B_sink_d(data);
}

#endif /* OMITGOOD */

} // close namespace
