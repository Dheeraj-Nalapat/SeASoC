/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE482_Comparing_instead_of_Assigning__basic_18.c
Label Definition File: CWE482_Comparing_instead_of_Assigning__basic.label.xml
Template File: point-flaw-18.tmpl.c
*/
/*
 * @description
 * CWE: 482 Comparing instead of Assigning
 * Sinks:
 *    GoodSink: Assigning
 *    BadSink : Comparing instead of assigning
 * Flow Variant: 18 Control flow: goto statements
 *
 * */

#include "std_testcase.h"

#ifndef OMITBAD

void CWE482_Comparing_instead_of_Assigning__basic_18_bad()
{
    goto sink;
    /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
    {
        int i = 0;
        /* FIX: use = instead of ==.  INCIDENTAL CWE-563: Unused Variable since the initial value of i (0) is never used */
        i = 5;
        printIntLine(i);
    }
sink:
    {
        int i = 0;
        /* FLAW: meant to assign 5 to i.  Maintenance note: may generate a compiler warning, this is intentional */
        i == 5;
        printIntLine(i);
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* good1() reverses the blocks on the goto statement */
static void good1()
{
    goto sink;
    /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
    {
        int i = 0;
        /* FLAW: meant to assign 5 to i.  Maintenance note: may generate a compiler warning, this is intentional */
        i == 5;
        printIntLine(i);
    }
sink:
    {
        int i = 0;
        /* FIX: use = instead of ==.  INCIDENTAL CWE-563: Unused Variable since the initial value of i (0) is never used */
        i = 5;
        printIntLine(i);
    }
}

void CWE482_Comparing_instead_of_Assigning__basic_18_good()
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
    CWE482_Comparing_instead_of_Assigning__basic_18_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE482_Comparing_instead_of_Assigning__basic_18_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
