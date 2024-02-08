/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE253_Incorrect_Check_of_Return_Value__wchar_t_remove_11.c
Label Definition File: CWE253_Incorrect_Check_of_Return_Value.string.label.xml
Template File: point-flaw-11.tmpl.c
*/
/*
 * @description
 * CWE: 253 Incorrect Check of Return Value
 * Sinks: remove
 *    GoodSink: Correctly check if wremove() failed
 *    BadSink : Incorrectly check if wremove() failed
 * Flow Variant: 11 Control flow: if(global_returns_t()) and if(global_returns_f())
 *
 * */

#include "std_testcase.h"

#ifdef _WIN32
# define REMOVE _wremove
#else
# define REMOVE wremove
#endif

#ifndef OMITBAD

void CWE253_Incorrect_Check_of_Return_Value__wchar_t_remove_11_bad()
{
    if(global_returns_t())
    {
        {
            /* FLAW: wremove() might fail, in which case the return value will be non-zero, but
             * we are checking to see if the return value is 0 */
            if (REMOVE(L"removemebad.txt") == 0)
            {
                printLine("wremove failed!");
            }
        }
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            /* FIX: check for the correct return value */
            if (REMOVE(L"removemebad.txt") != 0)
            {
                printLine("wremove failed!");
            }
        }
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* good1() uses if(global_returns_f()) instead of if(global_returns_t()) */
static void good1()
{
    if(global_returns_f())
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            /* FLAW: wremove() might fail, in which case the return value will be non-zero, but
             * we are checking to see if the return value is 0 */
            if (REMOVE(L"removemebad.txt") == 0)
            {
                printLine("wremove failed!");
            }
        }
    }
    else
    {
        {
            /* FIX: check for the correct return value */
            if (REMOVE(L"removemebad.txt") != 0)
            {
                printLine("wremove failed!");
            }
        }
    }
}

/* good2() reverses the bodies in the if statement */
static void good2()
{
    if(global_returns_t())
    {
        {
            /* FIX: check for the correct return value */
            if (REMOVE(L"removemebad.txt") != 0)
            {
                printLine("wremove failed!");
            }
        }
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            /* FLAW: wremove() might fail, in which case the return value will be non-zero, but
             * we are checking to see if the return value is 0 */
            if (REMOVE(L"removemebad.txt") == 0)
            {
                printLine("wremove failed!");
            }
        }
    }
}

void CWE253_Incorrect_Check_of_Return_Value__wchar_t_remove_11_good()
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
    CWE253_Incorrect_Check_of_Return_Value__wchar_t_remove_11_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE253_Incorrect_Check_of_Return_Value__wchar_t_remove_11_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
