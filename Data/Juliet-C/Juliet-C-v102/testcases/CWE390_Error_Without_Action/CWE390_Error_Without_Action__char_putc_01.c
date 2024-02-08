/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE390_Error_Without_Action__char_putc_01.c
Label Definition File: CWE390_Error_Without_Action.string.label.xml
Template File: point-flaw-01.tmpl.c
*/
/*
 * @description
 * CWE: 390 Detection of Error Condition Without Action
 * Sinks: putc
 *    GoodSink: Check if putc() failed and handle errors properly
 *    BadSink : Check to see if putc() failed, but do nothing about it
 * Flow Variant: 01 Baseline
 *
 * */

#include "std_testcase.h"

#ifndef OMITBAD

void CWE390_Error_Without_Action__char_putc_01_bad()
{
    {
        /* FLAW: check the return value, but do nothing if there is an error */
        if (putc((int)'A', stdout) == EOF)
        {
            /* do nothing */
        }
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

static void good1()
{
    {
        /* FIX: check the return value and handle errors properly */
        if (putc((int)'A', stdout) == EOF)
        {
            printLine("putc failed!");
            exit(1);
        }
    }
}

void CWE390_Error_Without_Action__char_putc_01_good()
{
    good1();
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
    CWE390_Error_Without_Action__char_putc_01_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE390_Error_Without_Action__char_putc_01_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
