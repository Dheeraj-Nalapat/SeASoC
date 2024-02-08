/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE121_Stack_Based_Buffer_Overflow__src_wchar_t_declare_snprintf_31.c
Label Definition File: CWE121_Stack_Based_Buffer_Overflow__src.string.label.xml
Template File: sources-sink-31.tmpl.c
*/
/*
 * @description
 * CWE: 121 Stack Based Buffer Overflow
 * BadSource:  Initialize data as a large string
 * GoodSource: Initialize data as a small string
 * Sinks: snprintf
 *    BadSink : Copy data to string using snwprintf
 * Flow Variant: 31 Data flow using a copy of data within the same function
 *
 * */

#include "std_testcase.h"

#include <wchar.h>

#ifndef OMITBAD

void CWE121_Stack_Based_Buffer_Overflow__src_wchar_t_declare_snprintf_31_bad()
{
    wchar_t * data;
    wchar_t data_buf[100];
    data = data_buf;
    /* FLAW: Initialize data as a large buffer that is larger than the small buffer used in the sink */
    wmemset(data, L'A', 100-1); /* fill with L'A's */
    data[100-1] = L'\0'; /* null terminate */
    {
        wchar_t * data_copy = data;
        wchar_t * data = data_copy;
        {
            wchar_t dest[50] = L"";
            /* POTENTIAL FLAW: Possible buffer overflow if data is larger than dest */
            _snwprintf(dest, wcslen(data), L"%s", data);
            printWLine(data);
        }
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B() uses the GoodSource with the BadSink */
static void goodG2B()
{
    wchar_t * data;
    wchar_t data_buf[100];
    data = data_buf;
    /* FIX: Initialize data as a small buffer that as small or smaller than the small buffer used in the sink */
    wmemset(data, L'A', 50-1); /* fill with L'A's */
    data[50-1] = L'\0'; /* null terminate */
    {
        wchar_t * data_copy = data;
        wchar_t * data = data_copy;
        {
            wchar_t dest[50] = L"";
            /* POTENTIAL FLAW: Possible buffer overflow if data is larger than dest */
            _snwprintf(dest, wcslen(data), L"%s", data);
            printWLine(data);
        }
    }
}

void CWE121_Stack_Based_Buffer_Overflow__src_wchar_t_declare_snprintf_31_good()
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
    CWE121_Stack_Based_Buffer_Overflow__src_wchar_t_declare_snprintf_31_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE121_Stack_Based_Buffer_Overflow__src_wchar_t_declare_snprintf_31_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif