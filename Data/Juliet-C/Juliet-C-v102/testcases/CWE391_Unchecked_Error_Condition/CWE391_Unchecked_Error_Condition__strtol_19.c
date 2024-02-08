/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE391_Unchecked_Error_Condition__strtol_19.c
Label Definition File: CWE391_Unchecked_Error_Condition.label.xml
Template File: point-flaw-19.tmpl.c
*/
/*
 * @description
 * CWE: 391 Unchecked Error Condition
 * Sinks: strtol
 *    GoodSink: Perform error checks and handling
 *    BadSink : Do not check if strtol() failed
 * Flow Variant: 19 Control flow: Dead code after a return
 *
 * */

#include "std_testcase.h"

#include <errno.h>

#ifndef OMITBAD

void CWE391_Unchecked_Error_Condition__strtol_19_bad()
{
    {
        long l = strtol("0xfffffffff", NULL, 0);
        /* FLAW: Do not check to see if strtol() failed */
        printf("%li\n", l);
    }
    return;
    /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
    {
        errno_t err_code = -1;
        long l = strtol("0xfffffffff", NULL, 0);
        if (_get_errno(&err_code))
        {
            printLine("_get_errno failed");
            exit(1);
        }
        /* FIX: Check errno to see if strtol() failed */
        if (err_code == ERANGE)
        {
            printLine("strtol() failed");
            exit(1);
        }
        printf("%li\n", l);
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* good1() reverses the GoodSinkBody and the BadSinkBody so that the BadSinkBody never runs */
static void good1()
{
    {
        errno_t err_code = -1;
        long l = strtol("0xfffffffff", NULL, 0);
        if (_get_errno(&err_code))
        {
            printLine("_get_errno failed");
            exit(1);
        }
        /* FIX: Check errno to see if strtol() failed */
        if (err_code == ERANGE)
        {
            printLine("strtol() failed");
            exit(1);
        }
        printf("%li\n", l);
    }
    return;
    /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
    {
        long l = strtol("0xfffffffff", NULL, 0);
        /* FLAW: Do not check to see if strtol() failed */
        printf("%li\n", l);
    }
}

void CWE391_Unchecked_Error_Condition__strtol_19_good()
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
    CWE391_Unchecked_Error_Condition__strtol_19_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE391_Unchecked_Error_Condition__strtol_19_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
