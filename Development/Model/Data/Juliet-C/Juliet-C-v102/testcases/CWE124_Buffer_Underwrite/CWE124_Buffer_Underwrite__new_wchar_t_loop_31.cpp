/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE124_Buffer_Underwrite__new_wchar_t_loop_31.cpp
Label Definition File: CWE124_Buffer_Underwrite__new.label.xml
Template File: sources-sink-31.tmpl.cpp
*/
/*
 * @description
 * CWE: 124 Buffer Underwrite
 * BadSource:  Set data pointer to before the allocated memory buffer
 * GoodSource: Set data pointer to the allocated memory buffer
 * Sinks: loop
 *    BadSink : Copy string to data using a loop
 * Flow Variant: 31 Data flow using a copy of data within the same function
 *
 * */

#include "std_testcase.h"

#include <wchar.h>

namespace CWE124_Buffer_Underwrite__new_wchar_t_loop_31
{

#ifndef OMITBAD

void bad()
{
    wchar_t * data;
    data = NULL;
    {
        wchar_t * data_buf = new wchar_t[100];
        wmemset(data_buf, L'A', 100-1);
        data_buf[100-1] = L'\0';
        /* FLAW: Set data pointer to before the allocated memory buffer */
        data = data_buf - 8;
    }
    {
        wchar_t * data_copy = data;
        wchar_t * data = data_copy;
        {
            size_t i;
            wchar_t src[100];
            wmemset(src, L'C', 100-1); /* fill with 'C's */
            src[100-1] = L'\0'; /* null terminate */
            /* POTENTIAL FLAW: Possibly copying data to memory before the destination buffer */
            for (i = 0; i < 100; i++)
            {
                data[i] = src[i];
            }
            /* Ensure the destination buffer is null terminated */
            data[100-1] = L'\0';
            printWLine(data);
            /* INCIDENTAL CWE-401: Memory Leak - data may not point to location
               returned by new [] so can't safely call delete [] on it */
        }
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B() uses the GoodSource with the BadSink */
static void goodG2B()
{
    wchar_t * data;
    data = NULL;
    {
        wchar_t * data_buf = new wchar_t[100];
        wmemset(data_buf, L'A', 100-1);
        data_buf[100-1] = L'\0';
        /* FIX: Set data pointer to the allocated memory buffer */
        data = data_buf;
    }
    {
        wchar_t * data_copy = data;
        wchar_t * data = data_copy;
        {
            size_t i;
            wchar_t src[100];
            wmemset(src, L'C', 100-1); /* fill with 'C's */
            src[100-1] = L'\0'; /* null terminate */
            /* POTENTIAL FLAW: Possibly copying data to memory before the destination buffer */
            for (i = 0; i < 100; i++)
            {
                data[i] = src[i];
            }
            /* Ensure the destination buffer is null terminated */
            data[100-1] = L'\0';
            printWLine(data);
            /* INCIDENTAL CWE-401: Memory Leak - data may not point to location
               returned by new [] so can't safely call delete [] on it */
        }
    }
}

void good()
{
    goodG2B();
}

#endif /* OMITGOOD */

} // close namespace

/* Below is the main(). It is only used when building this testcase on
   its own for testing or for building a binary to use in testing binary
   analysis tools. It is not used when compiling all the testcases as one
   application, which is how source code analysis tools are tested. */
#ifdef INCLUDEMAIN

using namespace CWE124_Buffer_Underwrite__new_wchar_t_loop_31; // so that we can use good and bad easily

int main(int argc, char * argv[])
{
    /* seed randomness */
    srand( (unsigned)time(NULL) );
#ifndef OMITGOOD
    printLine("Calling good()...");
    good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
