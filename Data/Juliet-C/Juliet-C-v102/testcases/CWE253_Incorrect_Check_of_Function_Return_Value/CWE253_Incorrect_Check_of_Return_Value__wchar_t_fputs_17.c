/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE253_Incorrect_Check_of_Return_Value__wchar_t_fputs_17.c
Label Definition File: CWE253_Incorrect_Check_of_Return_Value.string.label.xml
Template File: point-flaw-17.tmpl.c
*/
/*
 * @description
 * CWE: 253 Incorrect Check of Return Value
 * Sinks: fputs
 *    GoodSink: Correctly check if fputws() failed
 *    BadSink : Incorrectly check if fputws() failed
 * Flow Variant: 17 Control flow: for loops
 *
 * */

#include "std_testcase.h"

#ifndef OMITBAD

void CWE253_Incorrect_Check_of_Return_Value__wchar_t_fputs_17_bad()
{
    int j,k;
    for(j = 0; j < 1; j++)
    {
        {
            /* FLAW: fputws() might fail, in which case the return value will be WEOF (-1), but
             * we are checking to see if the return value is 0 */
            if (fputws(L"string", stdout) == 0)
            {
                printLine("fputws failed!");
            }
        }
    }
    for(k = 0; k < 0; k++)
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            /* FIX: check for the correct return value */
            if (fputws(L"string", stdout) == WEOF)
            {
                printLine("fputws failed!");
            }
        }
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* good1() changes the conditions on the for statements */
static void good1()
{
    int j,k;
    for(j = 0; j < 0; j++)
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            /* FLAW: fputws() might fail, in which case the return value will be WEOF (-1), but
             * we are checking to see if the return value is 0 */
            if (fputws(L"string", stdout) == 0)
            {
                printLine("fputws failed!");
            }
        }
    }
    for(k = 0; k < 1; k++)
    {
        {
            /* FIX: check for the correct return value */
            if (fputws(L"string", stdout) == WEOF)
            {
                printLine("fputws failed!");
            }
        }
    }
}

void CWE253_Incorrect_Check_of_Return_Value__wchar_t_fputs_17_good()
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
    CWE253_Incorrect_Check_of_Return_Value__wchar_t_fputs_17_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE253_Incorrect_Check_of_Return_Value__wchar_t_fputs_17_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
