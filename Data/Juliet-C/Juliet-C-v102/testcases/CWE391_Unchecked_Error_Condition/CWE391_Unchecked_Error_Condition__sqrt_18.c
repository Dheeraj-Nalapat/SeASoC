/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE391_Unchecked_Error_Condition__sqrt_18.c
Label Definition File: CWE391_Unchecked_Error_Condition.label.xml
Template File: point-flaw-18.tmpl.c
*/
/*
 * @description
 * CWE: 391 Unchecked Error Condition
 * Sinks: sqrt
 *    GoodSink: Perform error checks and handling
 *    BadSink : Do not check if sqrt() failed
 * Flow Variant: 18 Control flow: goto statements
 *
 * */

#include "std_testcase.h"

#include <errno.h>

#include <math.h>

#ifndef OMITBAD

void CWE391_Unchecked_Error_Condition__sqrt_18_bad()
{
    goto sink;
    /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
    {
        errno_t err_code = -1;
        double d = (double)sqrt((double)-1);
        if (_get_errno(&err_code))
        {
            printLine("_get_errno failed");
            exit(1);
        }
        /* FIX: Check errno to see if sqrt() failed */
        if (err_code == EDOM)
        {
            printLine("sqrt() failed");
            exit(1);
        }
        printDoubleLine(d);
    }
sink:
    {
        double d = (double)sqrt((double)-1);
        /* FLAW: Do not check to see if sqrt() failed */
        printDoubleLine(d);
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
        double d = (double)sqrt((double)-1);
        /* FLAW: Do not check to see if sqrt() failed */
        printDoubleLine(d);
    }
sink:
    {
        errno_t err_code = -1;
        double d = (double)sqrt((double)-1);
        if (_get_errno(&err_code))
        {
            printLine("_get_errno failed");
            exit(1);
        }
        /* FIX: Check errno to see if sqrt() failed */
        if (err_code == EDOM)
        {
            printLine("sqrt() failed");
            exit(1);
        }
        printDoubleLine(d);
    }
}

void CWE391_Unchecked_Error_Condition__sqrt_18_good()
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
    CWE391_Unchecked_Error_Condition__sqrt_18_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE391_Unchecked_Error_Condition__sqrt_18_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
