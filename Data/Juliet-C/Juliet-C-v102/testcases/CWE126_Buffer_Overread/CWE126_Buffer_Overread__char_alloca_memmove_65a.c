/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE126_Buffer_Overread__char_alloca_memmove_65a.c
Label Definition File: CWE126_Buffer_Overread.stack.label.xml
Template File: sources-sink-65a.tmpl.c
*/
/*
 * @description
 * CWE: 126 Buffer Over-read
 * BadSource:  Set data pointer to a small buffer
 * GoodSource: Set data pointer to a large buffer
 * Sinks: memmove
 *    BadSink : Copy data to string using memmove
 * Flow Variant: 65 Data/control flow: data passed as an argument from one function to a function in a different source file called via a function pointer
 *
 * */

#include "std_testcase.h"

#include <wchar.h>

#ifndef OMITBAD

/* bad function declaration */
void CWE126_Buffer_Overread__char_alloca_memmove_65b_bad_sink(char * data);

void CWE126_Buffer_Overread__char_alloca_memmove_65_bad()
{
    char * data;
    /* define a function pointer */
    void (*func_ptr) (char *) = CWE126_Buffer_Overread__char_alloca_memmove_65b_bad_sink;
    char * data_badbuf = (char *)ALLOCA(50*sizeof(char));
    char * data_goodbuf = (char *)ALLOCA(100*sizeof(char));
    memset(data_badbuf, 'A', 50-1); /* fill with 'A's */
    data_badbuf[50-1] = '\0'; /* null terminate */
    memset(data_goodbuf, 'A', 100-1); /* fill with 'A's */
    data_goodbuf[100-1] = '\0'; /* null terminate */
    /* FLAW: Set data pointer to a small buffer */
    data = data_badbuf;
    /* use the function pointer */
    func_ptr(data);
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B uses the GoodSource with the BadSink */
void CWE126_Buffer_Overread__char_alloca_memmove_65b_goodG2B_sink(char * data);

static void goodG2B()
{
    char * data;
    void (*func_ptr) (char *) = CWE126_Buffer_Overread__char_alloca_memmove_65b_goodG2B_sink;
    char * data_badbuf = (char *)ALLOCA(50*sizeof(char));
    char * data_goodbuf = (char *)ALLOCA(100*sizeof(char));
    memset(data_badbuf, 'A', 50-1); /* fill with 'A's */
    data_badbuf[50-1] = '\0'; /* null terminate */
    memset(data_goodbuf, 'A', 100-1); /* fill with 'A's */
    data_goodbuf[100-1] = '\0'; /* null terminate */
    /* FIX: Set data pointer to a large buffer */
    data = data_goodbuf;
    func_ptr(data);
}

void CWE126_Buffer_Overread__char_alloca_memmove_65_good()
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
    CWE126_Buffer_Overread__char_alloca_memmove_65_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE126_Buffer_Overread__char_alloca_memmove_65_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
