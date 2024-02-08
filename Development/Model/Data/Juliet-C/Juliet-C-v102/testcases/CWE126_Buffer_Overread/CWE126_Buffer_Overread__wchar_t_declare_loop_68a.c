/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE126_Buffer_Overread__wchar_t_declare_loop_68a.c
Label Definition File: CWE126_Buffer_Overread.stack.label.xml
Template File: sources-sink-68a.tmpl.c
*/
/*
 * @description
 * CWE: 126 Buffer Over-read
 * BadSource:  Set data pointer to a small buffer
 * GoodSource: Set data pointer to a large buffer
 * Sink: loop
 *    BadSink : Copy data to string using a loop
 * Flow Variant: 68 Data flow: data passed as a global variable from one function to another in different source files
 *
 * */

#include "std_testcase.h"

#include <wchar.h>

wchar_t * CWE126_Buffer_Overread__wchar_t_declare_loop_68_bad_data;
wchar_t * CWE126_Buffer_Overread__wchar_t_declare_loop_68_goodG2B_data;

#ifndef OMITBAD

/* bad function declaration */
void CWE126_Buffer_Overread__wchar_t_declare_loop_68b_bad_sink();

void CWE126_Buffer_Overread__wchar_t_declare_loop_68_bad()
{
    wchar_t * data;
    wchar_t data_badbuf[50];
    wchar_t data_goodbuf[100];
    wmemset(data_badbuf, L'A', 50-1); /* fill with 'A's */
    data_badbuf[50-1] = L'\0'; /* null terminate */
    wmemset(data_goodbuf, L'A', 100-1); /* fill with 'A's */
    data_goodbuf[100-1] = L'\0'; /* null terminate */
    /* FLAW: Set data pointer to a small buffer */
    data = data_badbuf;
    CWE126_Buffer_Overread__wchar_t_declare_loop_68_bad_data = data;
    CWE126_Buffer_Overread__wchar_t_declare_loop_68b_bad_sink();
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* good function declarations */
void CWE126_Buffer_Overread__wchar_t_declare_loop_68b_goodG2B_sink();

/* goodG2B uses the GoodSource with the BadSink */
static void goodG2B()
{
    wchar_t * data;
    wchar_t data_badbuf[50];
    wchar_t data_goodbuf[100];
    wmemset(data_badbuf, L'A', 50-1); /* fill with 'A's */
    data_badbuf[50-1] = L'\0'; /* null terminate */
    wmemset(data_goodbuf, L'A', 100-1); /* fill with 'A's */
    data_goodbuf[100-1] = L'\0'; /* null terminate */
    /* FIX: Set data pointer to a large buffer */
    data = data_goodbuf;
    CWE126_Buffer_Overread__wchar_t_declare_loop_68_goodG2B_data = data;
    CWE126_Buffer_Overread__wchar_t_declare_loop_68b_goodG2B_sink();
}

void CWE126_Buffer_Overread__wchar_t_declare_loop_68_good()
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
    CWE126_Buffer_Overread__wchar_t_declare_loop_68_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE126_Buffer_Overread__wchar_t_declare_loop_68_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
