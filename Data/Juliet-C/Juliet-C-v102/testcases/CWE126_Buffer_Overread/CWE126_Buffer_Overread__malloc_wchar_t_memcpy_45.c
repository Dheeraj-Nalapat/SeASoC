/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE126_Buffer_Overread__malloc_wchar_t_memcpy_45.c
Label Definition File: CWE126_Buffer_Overread__malloc.label.xml
Template File: sources-sink-45.tmpl.c
*/
/*
 * @description
 * CWE: 126 Buffer Over-read
 * BadSource:  Use a small buffer
 * GoodSource: Use a large buffer
 * Sinks: memcpy
 *    BadSink : Copy data to string using memcpy
 * Flow Variant: 45 Data flow: data passed as a static global variable from one function to another in the same source file
 *
 * */

#include "std_testcase.h"

#include <wchar.h>

static wchar_t * CWE126_Buffer_Overread__malloc_wchar_t_memcpy_45_bad_data;
static wchar_t * CWE126_Buffer_Overread__malloc_wchar_t_memcpy_45_goodG2B_data;

#ifndef OMITBAD

static void bad_sink()
{
    wchar_t * data = CWE126_Buffer_Overread__malloc_wchar_t_memcpy_45_bad_data;
    {
        wchar_t dest[100];
        wmemset(dest, L'C', 100-1);
        dest[100-1] = L'\0'; /* null terminate */
        /* POTENTIAL FLAW: using memcpy with the length of the dest where data
         * could be smaller than dest causing buffer overread */
        memcpy(dest, data, wcslen(dest)*sizeof(wchar_t));
        dest[100-1] = L'\0';
        printWLine(dest);
        free(data);
    }
}

void CWE126_Buffer_Overread__malloc_wchar_t_memcpy_45_bad()
{
    wchar_t * data;
    data = NULL;
    /* FLAW: Use a small buffer */
    data = (wchar_t *)malloc(50*sizeof(wchar_t));
    wmemset(data, L'A', 50-1); /* fill with 'A's */
    data[50-1] = L'\0'; /* null terminate */
    CWE126_Buffer_Overread__malloc_wchar_t_memcpy_45_bad_data = data;
    bad_sink();
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B() uses the GoodSource with the BadSink */
static void goodG2B_sink()
{
    wchar_t * data = CWE126_Buffer_Overread__malloc_wchar_t_memcpy_45_goodG2B_data;
    {
        wchar_t dest[100];
        wmemset(dest, L'C', 100-1);
        dest[100-1] = L'\0'; /* null terminate */
        /* POTENTIAL FLAW: using memcpy with the length of the dest where data
         * could be smaller than dest causing buffer overread */
        memcpy(dest, data, wcslen(dest)*sizeof(wchar_t));
        dest[100-1] = L'\0';
        printWLine(dest);
        free(data);
    }
}

static void goodG2B()
{
    wchar_t * data;
    data = NULL;
    /* FIX: Use a large buffer */
    data = (wchar_t *)malloc(100*sizeof(wchar_t));
    wmemset(data, L'A', 100-1); /* fill with 'A's */
    data[100-1] = L'\0'; /* null terminate */
    CWE126_Buffer_Overread__malloc_wchar_t_memcpy_45_goodG2B_data = data;
    goodG2B_sink();
}

void CWE126_Buffer_Overread__malloc_wchar_t_memcpy_45_good()
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
    CWE126_Buffer_Overread__malloc_wchar_t_memcpy_45_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE126_Buffer_Overread__malloc_wchar_t_memcpy_45_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
