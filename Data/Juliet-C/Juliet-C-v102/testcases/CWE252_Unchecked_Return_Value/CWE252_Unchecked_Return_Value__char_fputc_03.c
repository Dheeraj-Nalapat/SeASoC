/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE252_Unchecked_Return_Value__char_fputc_03.c
Label Definition File: CWE252_Unchecked_Return_Value.string.label.xml
Template File: point-flaw-03.tmpl.c
*/
/*
 * @description
 * CWE: 252 Unchecked Return Value
 * Sinks: fputc
 *    GoodSink: Check if fputc() fails
 *    BadSink : Do not check if fputc() fails
 * Flow Variant: 03 Control flow: if(5==5) and if(5!=5)
 *
 * */

#include "std_testcase.h"

#ifndef OMITBAD

void CWE252_Unchecked_Return_Value__char_fputc_03_bad()
{
    if(5==5)
    {
        {
            /* FLAW: Do not check the return value */
            fputc((int)'A', stdout);
        }
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            /* FIX: check the return value */
            if (fputc((int)'A', stdout) == EOF)
            {
                printLine("fputc failed!");
            }
        }
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* good1() uses if(5!=5) instead of if(5==5) */
static void good1()
{
    if(5!=5)
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            /* FLAW: Do not check the return value */
            fputc((int)'A', stdout);
        }
    }
    else
    {
        {
            /* FIX: check the return value */
            if (fputc((int)'A', stdout) == EOF)
            {
                printLine("fputc failed!");
            }
        }
    }
}

/* good2() reverses the bodies in the if statement */
static void good2()
{
    if(5==5)
    {
        {
            /* FIX: check the return value */
            if (fputc((int)'A', stdout) == EOF)
            {
                printLine("fputc failed!");
            }
        }
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            /* FLAW: Do not check the return value */
            fputc((int)'A', stdout);
        }
    }
}

void CWE252_Unchecked_Return_Value__char_fputc_03_good()
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
    CWE252_Unchecked_Return_Value__char_fputc_03_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE252_Unchecked_Return_Value__char_fputc_03_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
