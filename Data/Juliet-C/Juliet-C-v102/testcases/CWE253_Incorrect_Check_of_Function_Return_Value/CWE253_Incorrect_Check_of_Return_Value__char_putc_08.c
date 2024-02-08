/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE253_Incorrect_Check_of_Return_Value__char_putc_08.c
Label Definition File: CWE253_Incorrect_Check_of_Return_Value.string.label.xml
Template File: point-flaw-08.tmpl.c
*/
/*
 * @description
 * CWE: 253 Incorrect Check of Return Value
 * Sinks: putc
 *    GoodSink: Correctly check if putc() failed
 *    BadSink : Incorrectly check if putc() failed
 * Flow Variant: 08 Control flow: if(static_returns_t()) and if(static_returns_f())
 *
 * */

#include "std_testcase.h"

/* The two function below always return the same value, so a tool
   should be able to identify that calls to the functions will always
   return a fixed value. */
static int static_returns_t()
{
    return 1;
}

static int static_returns_f()
{
    return 0;
}

#ifndef OMITBAD

void CWE253_Incorrect_Check_of_Return_Value__char_putc_08_bad()
{
    if(static_returns_t())
    {
        {
            /* FLAW: putc() might fail, in which case the return value will be EOF (-1), but
             * we are checking to see if the return value is 0 */
            if (putc((int)'A', stdout) == 0)
            {
                printLine("putc failed!");
            }
        }
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            /* FIX: check for the correct return value */
            if (putc((int)'A', stdout) == EOF)
            {
                printLine("putc failed!");
            }
        }
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* good1() uses if(static_returns_f()) instead of if(static_returns_t()) */
static void good1()
{
    if(static_returns_f())
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            /* FLAW: putc() might fail, in which case the return value will be EOF (-1), but
             * we are checking to see if the return value is 0 */
            if (putc((int)'A', stdout) == 0)
            {
                printLine("putc failed!");
            }
        }
    }
    else
    {
        {
            /* FIX: check for the correct return value */
            if (putc((int)'A', stdout) == EOF)
            {
                printLine("putc failed!");
            }
        }
    }
}

/* good2() reverses the bodies in the if statement */
static void good2()
{
    if(static_returns_t())
    {
        {
            /* FIX: check for the correct return value */
            if (putc((int)'A', stdout) == EOF)
            {
                printLine("putc failed!");
            }
        }
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            /* FLAW: putc() might fail, in which case the return value will be EOF (-1), but
             * we are checking to see if the return value is 0 */
            if (putc((int)'A', stdout) == 0)
            {
                printLine("putc failed!");
            }
        }
    }
}

void CWE253_Incorrect_Check_of_Return_Value__char_putc_08_good()
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
    CWE253_Incorrect_Check_of_Return_Value__char_putc_08_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE253_Incorrect_Check_of_Return_Value__char_putc_08_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
