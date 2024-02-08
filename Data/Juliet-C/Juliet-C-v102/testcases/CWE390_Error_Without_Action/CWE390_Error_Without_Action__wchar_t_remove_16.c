/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE390_Error_Without_Action__wchar_t_remove_16.c
Label Definition File: CWE390_Error_Without_Action.string.label.xml
Template File: point-flaw-16.tmpl.c
*/
/*
 * @description
 * CWE: 390 Detection of Error Condition Without Action
 * Sinks: remove
 *    GoodSink: Check if wremove() failed and handle errors properly
 *    BadSink : Check to see if remove() failed, but do nothing about it
 * Flow Variant: 16 Control flow: while(1) and while(0)
 *
 * */

#include "std_testcase.h"

#ifdef _WIN32
# define REMOVE _wremove
#else
# define REMOVE wremove
#endif

#ifndef OMITBAD

void CWE390_Error_Without_Action__wchar_t_remove_16_bad()
{
    while(1)
    {
        {
            /* FLAW: check the return value, but do nothing if there is an error */
            if (REMOVE(L"removemebad.txt") != 0)
            {
                /* do nothing */
            }
        }
        break;
    }
    while(0)
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
        break;
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* good1() changes the conditions on the while statements */
static void good1()
{
    while(0)
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            /* FLAW: check the return value, but do nothing if there is an error */
            if (REMOVE(L"removemebad.txt") != 0)
            {
                /* do nothing */
            }
        }
        break;
    }
    while(1)
    {
        {
            /* FIX: check the return value and handle errors properly */
            if (REMOVE(L"removemebad.txt") != 0)
            {
                printLine("wremove failed!");
                exit(1);
            }
        }
        break;
    }
}

void CWE390_Error_Without_Action__wchar_t_remove_16_good()
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
    CWE390_Error_Without_Action__wchar_t_remove_16_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE390_Error_Without_Action__wchar_t_remove_16_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
