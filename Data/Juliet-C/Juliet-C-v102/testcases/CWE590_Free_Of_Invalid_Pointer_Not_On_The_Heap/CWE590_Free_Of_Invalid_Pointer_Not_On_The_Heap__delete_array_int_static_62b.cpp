/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE590_Free_Of_Invalid_Pointer_Not_On_The_Heap__delete_array_int_static_62b.cpp
Label Definition File: CWE590_Free_Of_Invalid_Pointer_Not_On_The_Heap__delete_array.label.xml
Template File: sources-sink-62b.tmpl.cpp
*/
/*
 * @description
 * CWE: 590 Free of Invalid Pointer Not on the Heap
 * BadSource: static Data buffer is declared static on the stack
 * GoodSource: Allocate memory on the heap
 * Sinks:
 *    BadSink : Print then free data
 * Flow Variant: 62 Data flow: data flows using a C++ reference from one function to another in different source files
 *
 * */

#include "std_testcase.h"

#include <wchar.h>

namespace CWE590_Free_Of_Invalid_Pointer_Not_On_The_Heap__delete_array_int_static_62
{

#ifndef OMITBAD

void bad_source(int * &data)
{
    {
        /* FLAW: data is allocated on the stack and deallocated in the BadSink */
        static int data_buf[100];
        {
            size_t i;
            for (i = 0; i < 100; i++)
            {
                data_buf[i] = 5;
            }
        }
        data = data_buf;
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B() uses the GoodSource with the BadSink */
void goodG2B_source(int * &data)
{
    {
        /* FIX: data is allocated on the heap and deallocated in the BadSink */
        int * data_buf = new int[100];
        {
            size_t i;
            for (i = 0; i < 100; i++)
            {
                data_buf[i] = 5;
            }
        }
        data = data_buf;
    }
}

#endif /* OMITGOOD */

} // close namespace
