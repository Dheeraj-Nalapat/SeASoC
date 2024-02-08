/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE390_Error_Without_Action__wchar_t_remove_14.c
Label Definition File: CWE390_Error_Without_Action.string.label.xml
Template File: point-flaw-14.tmpl.c
*/
/*
 * @description
 * CWE: 390 Detection of Error Condition Without Action
 * Sinks: remove
 *    GoodSink: Check if wremove() failed and handle errors properly
 *    BadSink : Check to see if remove() failed, but do nothing about it
 * Flow Variant: 14 Control flow: if(global_five==5) and if(global_five!=5)
 *
 * */

#include "std_testcase.h"

#ifdef _WIN32
# define REMOVE _wremove
#else
# define REMOVE wremove
#endif

#ifndef OMITBAD

void CWE390_Error_Without_Action__wchar_t_remove_14_bad()
{
    if(global_five==5)
    {
        {
            /* FLAW: check the return value, but do nothing if there is an error */
            if (REMOVE(L"removemebad.txt") != 0)
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
            if (REMOVE(L"removemebad.txt") != 0)
            {
                printLine("wremove failed!");
                exit(1);
            }
        }
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* good1() uses if(global_five!=5) instead of if(global_five==5) */
static void good1()
{
    if(global_five!=5)
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            /* FLAW: check the return value, but do nothing if there is an error */
            if (REMOVE(L"removemebad.txt") != 0)
            {
                /* do nothing */
            }
        }
    }
    else
    {
        {
            /* FIX: check the return value and handle errors properly */
            if (REMOVE(L"removemebad.txt") != 0)
            {
                printLine("wremove failed!");
                exit(1);
            }
        }
    }
}

/* good2() reverses the bodies in the if statement */
static void good2()
{
    if(global_five==5)
    {
        {
            /* FIX: check the return value and handle errors properly */
            if (REMOVE(L"removemebad.txt") != 0)
            {
                printLine("wremove failed!");
                exit(1);
            }
        }
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            /* FLAW: check the return value, but do nothing if there is an error */
            if (REMOVE(L"removemebad.txt") != 0)
            {
                /* do nothing */
            }
        }
    }
}

void CWE390_Error_Without_Action__wchar_t_remove_14_good()
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
    CWE390_Error_Without_Action__wchar_t_remove_14_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE390_Error_Without_Action__wchar_t_remove_14_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
