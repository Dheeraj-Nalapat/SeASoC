/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE253_Incorrect_Check_of_Return_Value__wchar_t_remove_01.c
Label Definition File: CWE253_Incorrect_Check_of_Return_Value.string.label.xml
Template File: point-flaw-01.tmpl.c
*/
/*
 * @description
 * CWE: 253 Incorrect Check of Return Value
 * Sinks: remove
 *    GoodSink: Correctly check if wremove() failed
 *    BadSink : Incorrectly check if wremove() failed
 * Flow Variant: 01 Baseline
 *
 * */

#include "std_testcase.h"

#ifdef _WIN32
# define REMOVE _wremove
#else
# define REMOVE wremove
#endif

#ifndef OMITBAD

void CWE253_Incorrect_Check_of_Return_Value__wchar_t_remove_01_bad()
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

#endif /* OMITBAD */

#ifndef OMITGOOD

static void good1()
{
    {
        /* FIX: check for the correct return value */
        if (REMOVE(L"removemebad.txt") != 0)
        {
            printLine("wremove failed!");
        }
    }
}

void CWE253_Incorrect_Check_of_Return_Value__wchar_t_remove_01_good()
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
    CWE253_Incorrect_Check_of_Return_Value__wchar_t_remove_01_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE253_Incorrect_Check_of_Return_Value__wchar_t_remove_01_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
