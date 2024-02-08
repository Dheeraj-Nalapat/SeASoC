/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE390_Error_Without_Action__wchar_t_fprintf_09.c
Label Definition File: CWE390_Error_Without_Action.string.label.xml
Template File: point-flaw-09.tmpl.c
*/
/*
 * @description
 * CWE: 390 Detection of Error Condition Without Action
 * Sinks: fprintf
 *    GoodSink: Check if fwprintf() failed and handle errors properly
 *    BadSink : Check to see if fwprintf() failed, but do nothing about it
 * Flow Variant: 09 Control flow: if(global_const_t) and if(global_const_f)
 *
 * */

#include "std_testcase.h"

#ifndef OMITBAD

void CWE390_Error_Without_Action__wchar_t_fprintf_09_bad()
{
    if(global_const_t)
    {
        {
            /* FLAW: check the return value, but do nothing if there is an error */
            if (fwprintf(stdout, L"%s\n", L"string") < 0)
            {
                /* do nothing */
            }
        }
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            /* FIX: check the return value and handle errors properly */
            if (fwprintf(stdout, L"%s\n", L"string") < 0)
            {
                printLine("fwprintf failed!");
                exit(1);
            }
        }
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* good1() uses if(global_const_f) instead of if(global_const_t) */
static void good1()
{
    if(global_const_f)
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            /* FLAW: check the return value, but do nothing if there is an error */
            if (fwprintf(stdout, L"%s\n", L"string") < 0)
            {
                /* do nothing */
            }
        }
    }
    else
    {
        {
            /* FIX: check the return value and handle errors properly */
            if (fwprintf(stdout, L"%s\n", L"string") < 0)
            {
                printLine("fwprintf failed!");
                exit(1);
            }
        }
    }
}

/* good2() reverses the bodies in the if statement */
static void good2()
{
    if(global_const_t)
    {
        {
            /* FIX: check the return value and handle errors properly */
            if (fwprintf(stdout, L"%s\n", L"string") < 0)
            {
                printLine("fwprintf failed!");
                exit(1);
            }
        }
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            /* FLAW: check the return value, but do nothing if there is an error */
            if (fwprintf(stdout, L"%s\n", L"string") < 0)
            {
                /* do nothing */
            }
        }
    }
}

void CWE390_Error_Without_Action__wchar_t_fprintf_09_good()
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
    CWE390_Error_Without_Action__wchar_t_fprintf_09_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE390_Error_Without_Action__wchar_t_fprintf_09_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
