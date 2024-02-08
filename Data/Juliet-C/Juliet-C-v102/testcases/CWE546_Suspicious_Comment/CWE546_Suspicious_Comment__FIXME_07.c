/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE546_Suspicious_Comment__FIXME_07.c
Label Definition File: CWE546_Suspicious_Comment.label.xml
Template File: point-flaw-07.tmpl.c
*/
/*
 * @description
 * CWE: 546 Suspicious Comment
 * Sinks: FIXME
 *    GoodSink: Comments show no indications of broken code
 *    BadSink : Comment contains the word FIXME
 * Flow Variant: 07 Control flow: if(static_five==5) and if(static_five!=5)
 *
 * */

#include "std_testcase.h"

/* The variable below is not declared "const", but is never assigned
   any other value so a tool should be able to identify that reads of
   this will always give its initialized value. */
static int static_five = 5;

#ifndef OMITBAD

void CWE546_Suspicious_Comment__FIXME_07_bad()
{
    if(static_five==5)
    {
        /* FLAW: The following comment has the letters 'FIXME' in it*/
        /* FIXME: This comment has the letters 'FIXME' in it, which is certainly
         * suspicious, because it could indicate this code needs to be fixed.
         */
        printLine("Hello");
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* FIX: Removed the suspicious comments */
        printLine("Hello");
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* good1() uses if(static_five!=5) instead of if(static_five==5) */
static void good1()
{
    if(static_five!=5)
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* FLAW: The following comment has the letters 'FIXME' in it*/
        /* FIXME: This comment has the letters 'FIXME' in it, which is certainly
         * suspicious, because it could indicate this code needs to be fixed.
         */
        printLine("Hello");
    }
    else
    {
        /* FIX: Removed the suspicious comments */
        printLine("Hello");
    }
}

/* good2() reverses the bodies in the if statement */
static void good2()
{
    if(static_five==5)
    {
        /* FIX: Removed the suspicious comments */
        printLine("Hello");
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* FLAW: The following comment has the letters 'FIXME' in it*/
        /* FIXME: This comment has the letters 'FIXME' in it, which is certainly
         * suspicious, because it could indicate this code needs to be fixed.
         */
        printLine("Hello");
    }
}

void CWE546_Suspicious_Comment__FIXME_07_good()
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
    CWE546_Suspicious_Comment__FIXME_07_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE546_Suspicious_Comment__FIXME_07_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
