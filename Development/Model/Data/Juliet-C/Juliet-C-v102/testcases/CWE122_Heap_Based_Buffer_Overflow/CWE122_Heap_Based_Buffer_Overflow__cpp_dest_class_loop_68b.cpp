/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE122_Heap_Based_Buffer_Overflow__cpp_dest_class_loop_68b.cpp
Label Definition File: CWE122_Heap_Based_Buffer_Overflow__cpp_dest.label.xml
Template File: sources-sink-68b.tmpl.cpp
*/
/*
 * @description
 * CWE: 122 Heap Based Buffer Overflow
 * BadSource:  Allocate using new[] and set data pointer to a small buffer
 * GoodSource: Allocate using new[] and set data pointer to a large buffer
 * Sink: loop
 *    BadSink : Copy twointsclass array to data using a loop
 * Flow Variant: 68 Data flow: data passed as a global variable from one function to another in different source files
 *
 * */

#include "std_testcase.h"

extern twointsclass * CWE122_Heap_Based_Buffer_Overflow__cpp_dest_class_loop_68_bad_data;
extern twointsclass * CWE122_Heap_Based_Buffer_Overflow__cpp_dest_class_loop_68_goodG2B_data;

namespace CWE122_Heap_Based_Buffer_Overflow__cpp_dest_class_loop_68
{

/* all the sinks are the same, we just want to know where the hit originated if a tool flags one */

#ifndef OMITBAD

void bad_sink()
{
    twointsclass * data = CWE122_Heap_Based_Buffer_Overflow__cpp_dest_class_loop_68_bad_data;
    {
        twointsclass src[100];
        {
            size_t i;
            /* Initialize array */
            for (i = 0; i < 100; i++)
            {
                src[i].a = 0;
                src[i].b = 0;
            }
        }
        {
            size_t i;
            /* POTENTIAL FLAW: Possible buffer overflow if data < 100 */
            for (i = 0; i < 100; i++)
            {
                data[i] = src[i];
            }
            printIntLine(data[0].a);
            delete [] data;
        }
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B uses the GoodSource with the BadSink */
void goodG2B_sink()
{
    twointsclass * data = CWE122_Heap_Based_Buffer_Overflow__cpp_dest_class_loop_68_goodG2B_data;
    {
        twointsclass src[100];
        {
            size_t i;
            /* Initialize array */
            for (i = 0; i < 100; i++)
            {
                src[i].a = 0;
                src[i].b = 0;
            }
        }
        {
            size_t i;
            /* POTENTIAL FLAW: Possible buffer overflow if data < 100 */
            for (i = 0; i < 100; i++)
            {
                data[i] = src[i];
            }
            printIntLine(data[0].a);
            delete [] data;
        }
    }
}

#endif /* OMITGOOD */

} // close namespace
