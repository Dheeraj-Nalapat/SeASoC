/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE674_Uncontrolled_Recursion__infinite_recursive_call_03.c
Label Definition File: CWE674_Uncontrolled_Recursion.label.xml
Template File: point-flaw-03.tmpl.c
*/
/*
 * @description
 * CWE: 674 Uncontrolled Recursion
 * Sinks: infinite_recursive_call
 *    GoodSink: The function has a lower bound thus preventing infinite recursion
 *    BadSink : Function does not have a bound causing infinite recursion
 * Flow Variant: 03 Control flow: if(5==5) and if(5!=5)
 *
 * */

#include "std_testcase.h"

static void helper_bad()
{
    /* FLAW: this function causes infinite recursion */
    helper_bad(); /* maintenance note: this may generate a warning, this is on purpose */
}

static void helper_good(unsigned level)
{
    /* FIX: provide lower-bound for recurssion stop */
    if (level == 0)
    {
        return;
    }
    helper_good(level - 1);
}

#ifndef OMITBAD

void CWE674_Uncontrolled_Recursion__infinite_recursive_call_03_bad()
{
    if(5==5)
    {
        helper_bad();
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        helper_good(5);
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
        helper_bad();
    }
    else
    {
        helper_good(5);
    }
}

/* good2() reverses the bodies in the if statement */
static void good2()
{
    if(5==5)
    {
        helper_good(5);
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        helper_bad();
    }
}

void CWE674_Uncontrolled_Recursion__infinite_recursive_call_03_good()
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
    CWE674_Uncontrolled_Recursion__infinite_recursive_call_03_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE674_Uncontrolled_Recursion__infinite_recursive_call_03_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
