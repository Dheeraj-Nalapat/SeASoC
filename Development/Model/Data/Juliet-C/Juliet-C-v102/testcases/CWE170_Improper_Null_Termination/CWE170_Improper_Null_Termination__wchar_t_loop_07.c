/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE170_Improper_Null_Termination__wchar_t_loop_07.c
Label Definition File: CWE170_Improper_Null_Termination.label.xml
Template File: point-flaw-07.tmpl.c
*/
/*
 * @description
 * CWE: 170 Improper Null Termination
 * Sinks: loop
 *    GoodSink: Copy a string using a loop with explicit null termination
 *    BadSink : Copy a string using a loop without explicit null termination
 * Flow Variant: 07 Control flow: if(static_five==5) and if(static_five!=5)
 *
 * */

#include "std_testcase.h"

#include <wchar.h>

/* The variable below is not declared "const", but is never assigned
   any other value so a tool should be able to identify that reads of
   this will always give its initialized value. */
static int static_five = 5;

#ifndef OMITBAD

void CWE170_Improper_Null_Termination__wchar_t_loop_07_bad()
{
    if(static_five==5)
    {
        {
            wchar_t src[150], dest[100];
            int i;
            /* Initialize src */
            wmemset(src, L'A', 149);
            src[149] = L'\0';
            for(i=0; i < 100; i++)
            {
                dest[i] = src[i];
            }
            /* FLAW: do not explicitly null terminate dest after the loop */
            printWLine(dest);
        }
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            wchar_t src[150], dest[100];
            int i;
            /* Initialize src */
            wmemset(src, L'A', 149);
            src[149] = L'\0';
            for(i=0; i < 100; i++)
            {
                dest[i] = src[i];
            }
            dest[99] = L'\0'; /* FIX: null terminate dest */
            printWLine(dest);
        }
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* good1() uses if(static_five!=5) instead of if(static_five==5) */
static void good1()
{
    if(static_five!=5)
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            wchar_t src[150], dest[100];
            int i;
            /* Initialize src */
            wmemset(src, L'A', 149);
            src[149] = L'\0';
            for(i=0; i < 100; i++)
            {
                dest[i] = src[i];
            }
            /* FLAW: do not explicitly null terminate dest after the loop */
            printWLine(dest);
        }
    }
    else
    {
        {
            wchar_t src[150], dest[100];
            int i;
            /* Initialize src */
            wmemset(src, L'A', 149);
            src[149] = L'\0';
            for(i=0; i < 100; i++)
            {
                dest[i] = src[i];
            }
            dest[99] = L'\0'; /* FIX: null terminate dest */
            printWLine(dest);
        }
    }
}

/* good2() reverses the bodies in the if statement */
static void good2()
{
    if(static_five==5)
    {
        {
            wchar_t src[150], dest[100];
            int i;
            /* Initialize src */
            wmemset(src, L'A', 149);
            src[149] = L'\0';
            for(i=0; i < 100; i++)
            {
                dest[i] = src[i];
            }
            dest[99] = L'\0'; /* FIX: null terminate dest */
            printWLine(dest);
        }
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            wchar_t src[150], dest[100];
            int i;
            /* Initialize src */
            wmemset(src, L'A', 149);
            src[149] = L'\0';
            for(i=0; i < 100; i++)
            {
                dest[i] = src[i];
            }
            /* FLAW: do not explicitly null terminate dest after the loop */
            printWLine(dest);
        }
    }
}

void CWE170_Improper_Null_Termination__wchar_t_loop_07_good()
{
    good1();
    good2();
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
    CWE170_Improper_Null_Termination__wchar_t_loop_07_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE170_Improper_Null_Termination__wchar_t_loop_07_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
