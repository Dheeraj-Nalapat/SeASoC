/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE121_Stack_Based_Buffer_Overflow__src_wchar_t_declare_memmove_68a.c
Label Definition File: CWE121_Stack_Based_Buffer_Overflow__src.string.label.xml
Template File: sources-sink-68a.tmpl.c
*/
/*
 * @description
 * CWE: 121 Stack Based Buffer Overflow
 * BadSource:  Initialize data as a large string
 * GoodSource: Initialize data as a small string
 * Sink: memmove
 *    BadSink : Copy data to string using memmove
 * Flow Variant: 68 Data flow: data passed as a global variable from one function to another in different source files
 *
 * */

#include "std_testcase.h"

#include <wchar.h>

wchar_t * CWE121_Stack_Based_Buffer_Overflow__src_wchar_t_declare_memmove_68_bad_data;
wchar_t * CWE121_Stack_Based_Buffer_Overflow__src_wchar_t_declare_memmove_68_goodG2B_data;

#ifndef OMITBAD

/* bad function declaration */
void CWE121_Stack_Based_Buffer_Overflow__src_wchar_t_declare_memmove_68b_bad_sink();

void CWE121_Stack_Based_Buffer_Overflow__src_wchar_t_declare_memmove_68_bad()
{
    wchar_t * data;
    wchar_t data_buf[100];
    data = data_buf;
    /* FLAW: Initialize data as a large buffer that is larger than the small buffer used in the sink */
    wmemset(data, L'A', 100-1); /* fill with L'A's */
    data[100-1] = L'\0'; /* null terminate */
    CWE121_Stack_Based_Buffer_Overflow__src_wchar_t_declare_memmove_68_bad_data = data;
    CWE121_Stack_Based_Buffer_Overflow__src_wchar_t_declare_memmove_68b_bad_sink();
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* good function declarations */
void CWE121_Stack_Based_Buffer_Overflow__src_wchar_t_declare_memmove_68b_goodG2B_sink();

/* goodG2B uses the GoodSource with the BadSink */
static void goodG2B()
{
    wchar_t * data;
    wchar_t data_buf[100];
    data = data_buf;
    /* FIX: Initialize data as a small buffer that as small or smaller than the small buffer used in the sink */
    wmemset(data, L'A', 50-1); /* fill with L'A's */
    data[50-1] = L'\0'; /* null terminate */
    CWE121_Stack_Based_Buffer_Overflow__src_wchar_t_declare_memmove_68_goodG2B_data = data;
    CWE121_Stack_Based_Buffer_Overflow__src_wchar_t_declare_memmove_68b_goodG2B_sink();
}

void CWE121_Stack_Based_Buffer_Overflow__src_wchar_t_declare_memmove_68_good()
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
    CWE121_Stack_Based_Buffer_Overflow__src_wchar_t_declare_memmove_68_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE121_Stack_Based_Buffer_Overflow__src_wchar_t_declare_memmove_68_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
