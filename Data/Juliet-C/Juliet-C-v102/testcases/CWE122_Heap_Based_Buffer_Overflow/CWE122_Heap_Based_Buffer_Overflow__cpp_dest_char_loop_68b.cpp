/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE122_Heap_Based_Buffer_Overflow__cpp_dest_char_loop_68b.cpp
Label Definition File: CWE122_Heap_Based_Buffer_Overflow__cpp_dest.string.label.xml
Template File: sources-sink-68b.tmpl.cpp
*/
/*
 * @description
 * CWE: 122 Heap Based Buffer Overflow
 * BadSource:  Allocate using new[] and set data pointer to a small buffer
 * GoodSource: Allocate using new[] and set data pointer to a large buffer
 * Sink: loop
 *    BadSink : Copy string to data using a loop
 * Flow Variant: 68 Data flow: data passed as a global variable from one function to another in different source files
 *
 * */

#include "std_testcase.h"

#include <wchar.h>

extern char * CWE122_Heap_Based_Buffer_Overflow__cpp_dest_char_loop_68_bad_data;
extern char * CWE122_Heap_Based_Buffer_Overflow__cpp_dest_char_loop_68_goodG2B_data;

namespace CWE122_Heap_Based_Buffer_Overflow__cpp_dest_char_loop_68
{

/* all the sinks are the same, we just want to know where the hit originated if a tool flags one */

#ifndef OMITBAD

void bad_sink()
{
    char * data = CWE122_Heap_Based_Buffer_Overflow__cpp_dest_char_loop_68_bad_data;
    {
        size_t i;
        char src[100];
        memset(src, 'C', 100-1); /* fill with 'C's */
        src[100-1] = '\0'; /* null terminate */
        /* POTENTIAL FLAW: Possible buffer overflow if src is larger than data */
        for (i = 0; i < 100; i++)
        {
            data[i] = src[i];
        }
        data[100-1] = '\0'; /* Ensure the destination buffer is null terminated */
        printLine(data);
        delete [] data;
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B uses the GoodSource with the BadSink */
void goodG2B_sink()
{
    char * data = CWE122_Heap_Based_Buffer_Overflow__cpp_dest_char_loop_68_goodG2B_data;
    {
        size_t i;
        char src[100];
        memset(src, 'C', 100-1); /* fill with 'C's */
        src[100-1] = '\0'; /* null terminate */
        /* POTENTIAL FLAW: Possible buffer overflow if src is larger than data */
        for (i = 0; i < 100; i++)
        {
            data[i] = src[i];
        }
        data[100-1] = '\0'; /* Ensure the destination buffer is null terminated */
        printLine(data);
        delete [] data;
    }
}

#endif /* OMITGOOD */

} // close namespace
