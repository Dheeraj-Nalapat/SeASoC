/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE252_Unchecked_Return_Value__wchar_t_puts_13.c
Label Definition File: CWE252_Unchecked_Return_Value.string.label.xml
Template File: point-flaw-13.tmpl.c
*/
/*
 * @description
 * CWE: 252 Unchecked Return Value
 * Sinks: puts
 *    GoodSink: Check if putws() fails
 *    BadSink : Do not check if _putws() fails
 * Flow Variant: 13 Control flow: if(global_const_five==5) and if(global_const_five!=5)
 *
 * */

#include "std_testcase.h"

#ifndef OMITBAD

void CWE252_Unchecked_Return_Value__wchar_t_puts_13_bad()
{
    if(global_const_five==5)
    {
        {
            /* FLAW: Do not check the return value */
            _putws(L"string");
        }
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            /* FIX: check the return value */
            if (_putws(L"string") == WEOF)
            {
                printLine("puts failed!");
            }
        }
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* good1() uses if(global_const_five!=5) instead of if(global_const_five==5) */
static void good1()
{
    if(global_const_five!=5)
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            /* FLAW: Do not check the return value */
            _putws(L"string");
        }
    }
    else
    {
        {
            /* FIX: check the return value */
            if (_putws(L"string") == WEOF)
            {
                printLine("puts failed!");
            }
        }
    }
}

/* good2() reverses the bodies in the if statement */
static void good2()
{
    if(global_const_five==5)
    {
        {
            /* FIX: check the return value */
            if (_putws(L"string") == WEOF)
            {
                printLine("puts failed!");
            }
        }
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            /* FLAW: Do not check the return value */
            _putws(L"string");
        }
    }
}

void CWE252_Unchecked_Return_Value__wchar_t_puts_13_good()
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
    CWE252_Unchecked_Return_Value__wchar_t_puts_13_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE252_Unchecked_Return_Value__wchar_t_puts_13_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif