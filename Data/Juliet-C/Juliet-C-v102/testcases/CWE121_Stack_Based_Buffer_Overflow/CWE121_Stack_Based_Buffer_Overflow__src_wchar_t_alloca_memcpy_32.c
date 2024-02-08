/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE121_Stack_Based_Buffer_Overflow__src_wchar_t_alloca_memcpy_32.c
Label Definition File: CWE121_Stack_Based_Buffer_Overflow__src.string.label.xml
Template File: sources-sink-32.tmpl.c
*/
/*
 * @description
 * CWE: 121 Stack Based Buffer Overflow
 * BadSource:  Initialize data as a large string
 * GoodSource: Initialize data as a small string
 * Sink: memcpy
 *    BadSink : Copy data to string using memcpy
 * Flow Variant: 32 Data flow using two pointers to the same value within the same function
 *
 * */

#include "std_testcase.h"

#include <wchar.h>

#ifndef OMITBAD

void CWE121_Stack_Based_Buffer_Overflow__src_wchar_t_alloca_memcpy_32_bad()
{
    wchar_t * data;
    wchar_t * *data_ptr1 = &data;
    wchar_t * *data_ptr2 = &data;
    wchar_t * data_buf = (wchar_t *)ALLOCA(100*sizeof(wchar_t));
    data = data_buf;
    {
        wchar_t * data = *data_ptr1;
        /* FLAW: Initialize data as a large buffer that is larger than the small buffer used in the sink */
        wmemset(data, L'A', 100-1); /* fill with L'A's */
        data[100-1] = L'\0'; /* null terminate */
        *data_ptr1 = data;
    }
    {
        wchar_t * data = *data_ptr2;
        {
            wchar_t dest[50] = L"";
            /* POTENTIAL FLAW: Possible buffer overflow if data is larger than dest */
            memcpy(dest, data, wcslen(data)*sizeof(wchar_t));
            dest[50-1] = L'\0'; /* Ensure the destination buffer is null terminated */
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
    wchar_t * *data_ptr1 = &data;
    wchar_t * *data_ptr2 = &data;
    wchar_t * data_buf = (wchar_t *)ALLOCA(100*sizeof(wchar_t));
    data = data_buf;
    {
        wchar_t * data = *data_ptr1;
        /* FIX: Initialize data as a small buffer that as small or smaller than the small buffer used in the sink */
        wmemset(data, L'A', 50-1); /* fill with L'A's */
        data[50-1] = L'\0'; /* null terminate */
        *data_ptr1 = data;
    }
    {
        wchar_t * data = *data_ptr2;
        {
            wchar_t dest[50] = L"";
            /* POTENTIAL FLAW: Possible buffer overflow if data is larger than dest */
            memcpy(dest, data, wcslen(data)*sizeof(wchar_t));
            dest[50-1] = L'\0'; /* Ensure the destination buffer is null terminated */
            printWLine(data);
        }
    }
}

void CWE121_Stack_Based_Buffer_Overflow__src_wchar_t_alloca_memcpy_32_good()
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
    CWE121_Stack_Based_Buffer_Overflow__src_wchar_t_alloca_memcpy_32_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE121_Stack_Based_Buffer_Overflow__src_wchar_t_alloca_memcpy_32_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif