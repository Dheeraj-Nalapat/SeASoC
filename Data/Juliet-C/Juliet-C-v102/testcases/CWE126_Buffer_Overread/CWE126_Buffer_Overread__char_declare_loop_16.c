/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE126_Buffer_Overread__char_declare_loop_16.c
Label Definition File: CWE126_Buffer_Overread.stack.label.xml
Template File: sources-sink-16.tmpl.c
*/
/*
 * @description
 * CWE: 126 Buffer Over-read
 * BadSource:  Set data pointer to a small buffer
 * GoodSource: Set data pointer to a large buffer
 * Sink: loop
 *    BadSink : Copy data to string using a loop
 * Flow Variant: 16 Control flow: while(1) and while(0)
 *
 * */

#include "std_testcase.h"

#include <wchar.h>

#ifndef OMITBAD

void CWE126_Buffer_Overread__char_declare_loop_16_bad()
{
    char * data;
    char data_badbuf[50];
    char data_goodbuf[100];
    memset(data_badbuf, 'A', 50-1); /* fill with 'A's */
    data_badbuf[50-1] = '\0'; /* null terminate */
    memset(data_goodbuf, 'A', 100-1); /* fill with 'A's */
    data_goodbuf[100-1] = '\0'; /* null terminate */
    while(0)
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* FIX: Set data pointer to a large buffer */
        data = data_goodbuf;
        break;
    }
    while(1)
    {
        /* FLAW: Set data pointer to a small buffer */
        data = data_badbuf;
        break;
    }
    {
        size_t i, dest_sz;
        char dest[100];
        memset(dest, 'C', 100-1);
        dest[100-1] = '\0'; /* null terminate */
        dest_sz = strlen(dest);
        /* POTENTIAL FLAW: using length of the dest where data
         * could be smaller than dest causing buffer overread */
        for (i = 0; i < dest_sz; i++)
        {
            dest[i] = data[i];
        }
        dest[100-1] = '\0';
        printLine(dest);
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B() - use goodsource and badsink by changing the conditions on the while statements */
static void goodG2B()
{
    char * data;
    char data_badbuf[50];
    char data_goodbuf[100];
    memset(data_badbuf, 'A', 50-1); /* fill with 'A's */
    data_badbuf[50-1] = '\0'; /* null terminate */
    memset(data_goodbuf, 'A', 100-1); /* fill with 'A's */
    data_goodbuf[100-1] = '\0'; /* null terminate */
    while(1)
    {
        /* FIX: Set data pointer to a large buffer */
        data = data_goodbuf;
        break;
    }
    while(0)
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* FLAW: Set data pointer to a small buffer */
        data = data_badbuf;
        break;
    }
    {
        size_t i, dest_sz;
        char dest[100];
        memset(dest, 'C', 100-1);
        dest[100-1] = '\0'; /* null terminate */
        dest_sz = strlen(dest);
        /* POTENTIAL FLAW: using length of the dest where data
         * could be smaller than dest causing buffer overread */
        for (i = 0; i < dest_sz; i++)
        {
            dest[i] = data[i];
        }
        dest[100-1] = '\0';
        printLine(dest);
    }
}

void CWE126_Buffer_Overread__char_declare_loop_16_good()
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
    CWE126_Buffer_Overread__char_declare_loop_16_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE126_Buffer_Overread__char_declare_loop_16_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
