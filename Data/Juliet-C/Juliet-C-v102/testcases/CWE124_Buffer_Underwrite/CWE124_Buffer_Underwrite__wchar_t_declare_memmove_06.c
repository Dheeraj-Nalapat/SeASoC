/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE124_Buffer_Underwrite__wchar_t_declare_memmove_06.c
Label Definition File: CWE124_Buffer_Underwrite.stack.label.xml
Template File: sources-sink-06.tmpl.c
*/
/*
 * @description
 * CWE: 124 Buffer Underwrite
 * BadSource:  Set data pointer to before the allocated memory buffer
 * GoodSource: Set data pointer to the allocated memory buffer
 * Sink: memmove
 *    BadSink : Copy string to data using memmove
 * Flow Variant: 06 Control flow: if(static_const_five==5) and if(static_const_five!=5)
 *
 * */

#include "std_testcase.h"

#include <wchar.h>

/* The variable below is declared "const", so a tool should be able
   to identify that reads of this will always give its initialized
   value. */
static const int static_const_five = 5;

#ifndef OMITBAD

void CWE124_Buffer_Underwrite__wchar_t_declare_memmove_06_bad()
{
    wchar_t * data;
    wchar_t data_buf[100];
    wmemset(data_buf, L'A', 100-1);
    data_buf[100-1] = L'\0';
    if(static_const_five==5)
    {
        /* FLAW: Set data pointer to before the allocated memory buffer */
        data = data_buf - 8;
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* FIX: Set data pointer to the allocated memory buffer */
        data = data_buf;
    }
    {
        wchar_t src[100];
        wmemset(src, L'C', 100-1); /* fill with 'C's */
        src[100-1] = L'\0'; /* null terminate */
        /* POTENTIAL FLAW: Possibly copying data to memory before the destination buffer */
        memmove(data, src, 100*sizeof(wchar_t));
        /* Ensure the destination buffer is null terminated */
        data[100-1] = L'\0';
        printWLine(data);
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B1() - use goodsource and badsink by changing the static_const_five==5 to static_const_five!=5 */
static void goodG2B1()
{
    wchar_t * data;
    wchar_t data_buf[100];
    wmemset(data_buf, L'A', 100-1);
    data_buf[100-1] = L'\0';
    if(static_const_five!=5)
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* FLAW: Set data pointer to before the allocated memory buffer */
        data = data_buf - 8;
    }
    else
    {
        /* FIX: Set data pointer to the allocated memory buffer */
        data = data_buf;
    }
    {
        wchar_t src[100];
        wmemset(src, L'C', 100-1); /* fill with 'C's */
        src[100-1] = L'\0'; /* null terminate */
        /* POTENTIAL FLAW: Possibly copying data to memory before the destination buffer */
        memmove(data, src, 100*sizeof(wchar_t));
        /* Ensure the destination buffer is null terminated */
        data[100-1] = L'\0';
        printWLine(data);
    }
}

/* goodG2B2() - use goodsource and badsink by reversing the blocks in the if statement */
static void goodG2B2()
{
    wchar_t * data;
    wchar_t data_buf[100];
    wmemset(data_buf, L'A', 100-1);
    data_buf[100-1] = L'\0';
    if(static_const_five==5)
    {
        /* FIX: Set data pointer to the allocated memory buffer */
        data = data_buf;
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* FLAW: Set data pointer to before the allocated memory buffer */
        data = data_buf - 8;
    }
    {
        wchar_t src[100];
        wmemset(src, L'C', 100-1); /* fill with 'C's */
        src[100-1] = L'\0'; /* null terminate */
        /* POTENTIAL FLAW: Possibly copying data to memory before the destination buffer */
        memmove(data, src, 100*sizeof(wchar_t));
        /* Ensure the destination buffer is null terminated */
        data[100-1] = L'\0';
        printWLine(data);
    }
}

void CWE124_Buffer_Underwrite__wchar_t_declare_memmove_06_good()
{
    goodG2B1();
    goodG2B2();
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
    CWE124_Buffer_Underwrite__wchar_t_declare_memmove_06_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE124_Buffer_Underwrite__wchar_t_declare_memmove_06_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif