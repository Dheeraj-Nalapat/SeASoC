/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE546_Suspicious_Comment__BUG_19.c
Label Definition File: CWE546_Suspicious_Comment.label.xml
Template File: point-flaw-19.tmpl.c
*/
/*
 * @description
 * CWE: 546 Suspicious Comment
 * Sinks: BUG
 *    GoodSink: Comments show no indications of bugs
 *    BadSink : Comment contains the word BUG
 * Flow Variant: 19 Control flow: Dead code after a return
 *
 * */

#include "std_testcase.h"

#ifndef OMITBAD

void CWE546_Suspicious_Comment__BUG_19_bad()
{
    /* FLAW: The following comment has the letters 'BUG' in it*/
    /* BUG: This comment has the letters 'BUG' in it, which is certainly
     * suspicious, because it could indicate this code has a BUG in it.
     */
    printLine("Hello");
    return;
    /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
    /* FIX: Removed the suspicious comments */
    printLine("Hello");
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* good1() reverses the GoodSinkBody and the BadSinkBody so that the BadSinkBody never runs */
static void good1()
{
    /* FIX: Removed the suspicious comments */
    printLine("Hello");
    return;
    /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
    /* FLAW: The following comment has the letters 'BUG' in it*/
    /* BUG: This comment has the letters 'BUG' in it, which is certainly
     * suspicious, because it could indicate this code has a BUG in it.
     */
    printLine("Hello");
}

void CWE546_Suspicious_Comment__BUG_19_good()
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
    CWE546_Suspicious_Comment__BUG_19_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE546_Suspicious_Comment__BUG_19_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
