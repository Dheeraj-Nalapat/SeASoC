/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE124_Buffer_Underwrite__new_char_cpy_65b.cpp
Label Definition File: CWE124_Buffer_Underwrite__new.label.xml
Template File: sources-sink-65b.tmpl.cpp
*/
/*
 * @description
 * CWE: 124 Buffer Underwrite
 * BadSource:  Set data pointer to before the allocated memory buffer
 * GoodSource: Set data pointer to the allocated memory buffer
 * Sinks: cpy
 *    BadSink : Copy string to data using strcpy
 * Flow Variant: 65 Data/control flow: data passed as an argument from one function to a function in a different source file called via a function pointer
 *
 * */

#include "std_testcase.h"

#include <wchar.h>

namespace CWE124_Buffer_Underwrite__new_char_cpy_65
{

#ifndef OMITBAD

void bad_sink(char * data)
{
    {
        char src[100];
        memset(src, 'C', 100-1); /* fill with 'C's */
        src[100-1] = '\0'; /* null terminate */
        /* POTENTIAL FLAW: Possibly copying data to memory before the destination buffer */
        strcpy(data, src);
        printLine(data);
        /* INCIDENTAL CWE-401: Memory Leak - data may not point to location
           returned by new [] so can't safely call delete [] on it */
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B uses the GoodSource with the BadSink */
void goodG2B_sink(char * data)
{
    {
        char src[100];
        memset(src, 'C', 100-1); /* fill with 'C's */
        src[100-1] = '\0'; /* null terminate */
        /* POTENTIAL FLAW: Possibly copying data to memory before the destination buffer */
        strcpy(data, src);
        printLine(data);
        /* INCIDENTAL CWE-401: Memory Leak - data may not point to location
           returned by new [] so can't safely call delete [] on it */
    }
}

#endif /* OMITGOOD */

} // close namespace