/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE124_Buffer_Underwrite__malloc_wchar_t_loop_31.c
Label Definition File: CWE124_Buffer_Underwrite__malloc.label.xml
Template File: sources-sink-31.tmpl.c
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

#ifndef OMITBAD

void CWE124_Buffer_Underwrite__malloc_wchar_t_loop_31_bad()
{
    wchar_t * data;
    data = NULL;
    {
        wchar_t * data_buf = (wchar_t *)malloc(100*sizeof(wchar_t));
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
               returned by malloc() so can't safely call free() on it */
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
        wchar_t * data_buf = (wchar_t *)malloc(100*sizeof(wchar_t));
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
               returned by malloc() so can't safely call free() on it */
        }
    }
}

void CWE124_Buffer_Underwrite__malloc_wchar_t_loop_31_good()
{
    goodG2B();
}

#endif /* OMITGOOD */

/* Below is the main(). It is only used when building this testcase on
   its own for testing or for building a binary to use in testing binary
   analysis tools. It is not used when compiling all the testcases as one
   application, which is how source code analysis tools are tested. */
#ifdef INCLUDEMAIN

int main(int argc, char * argv[])
{
    /* seed randomness */
    srand( (unsigned)time(NULL) );
#ifndef OMITGOOD
    printLine("Calling good()...");
    CWE124_Buffer_Underwrite__malloc_wchar_t_loop_31_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE124_Buffer_Underwrite__malloc_wchar_t_loop_31_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif