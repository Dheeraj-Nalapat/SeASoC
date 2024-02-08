/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE126_Buffer_Overread__char_declare_memmove_54a.c
Label Definition File: CWE126_Buffer_Overread.stack.label.xml
Template File: sources-sink-54a.tmpl.c
*/
/*
 * @description
 * CWE: 126 Buffer Over-read
 * BadSource:  Set data pointer to a small buffer
 * GoodSource: Set data pointer to a large buffer
 * Sink: memmove
 *    BadSink : Copy data to string using memmove
 * Flow Variant: 54 Data flow: data passed as an argument from one function through three others to a fifth; all five functions are in different source files
 *
 * */

#include "std_testcase.h"

#include <wchar.h>

#ifndef OMITBAD

/* bad function declaration */
void CWE126_Buffer_Overread__char_declare_memmove_54b_bad_sink(char * data);

void CWE126_Buffer_Overread__char_declare_memmove_54_bad()
{
    char * data;
    char data_badbuf[50];
    char data_goodbuf[100];
    memset(data_badbuf, 'A', 50-1); /* fill with 'A's */
    data_badbuf[50-1] = '\0'; /* null terminate */
    memset(data_goodbuf, 'A', 100-1); /* fill with 'A's */
    data_goodbuf[100-1] = '\0'; /* null terminate */
    /* FLAW: Set data pointer to a small buffer */
    data = data_badbuf;
    CWE126_Buffer_Overread__char_declare_memmove_54b_bad_sink(data);
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* good function declaration */
void CWE126_Buffer_Overread__char_declare_memmove_54b_goodG2B_sink(char * data);

/* goodG2B uses the GoodSource with the BadSink */
static void goodG2B()
{
    char * data;
    char data_badbuf[50];
    char data_goodbuf[100];
    memset(data_badbuf, 'A', 50-1); /* fill with 'A's */
    data_badbuf[50-1] = '\0'; /* null terminate */
    memset(data_goodbuf, 'A', 100-1); /* fill with 'A's */
    data_goodbuf[100-1] = '\0'; /* null terminate */
    /* FIX: Set data pointer to a large buffer */
    data = data_goodbuf;
    CWE126_Buffer_Overread__char_declare_memmove_54b_goodG2B_sink(data);
}

void CWE126_Buffer_Overread__char_declare_memmove_54_good()
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
    CWE126_Buffer_Overread__char_declare_memmove_54_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE126_Buffer_Overread__char_declare_memmove_54_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
