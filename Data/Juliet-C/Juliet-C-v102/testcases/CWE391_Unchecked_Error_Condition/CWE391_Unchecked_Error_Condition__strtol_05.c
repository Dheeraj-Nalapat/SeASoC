/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE391_Unchecked_Error_Condition__strtol_05.c
Label Definition File: CWE391_Unchecked_Error_Condition.label.xml
Template File: point-flaw-05.tmpl.c
*/
/*
 * @description
 * CWE: 391 Unchecked Error Condition
 * Sinks: strtol
 *    GoodSink: Perform error checks and handling
 *    BadSink : Do not check if strtol() failed
 * Flow Variant: 05 Control flow: if(static_t) and if(static_f)
 *
 * */

#include "std_testcase.h"

#include <errno.h>

/* The two variables below are not defined as "const", but are never
   assigned any other value, so a tool should be able to identify that
   reads of these will always return their initialized values. */
static int static_t = 1; /* true */
static int static_f = 0; /* false */

#ifndef OMITBAD

void CWE391_Unchecked_Error_Condition__strtol_05_bad()
{
    if(static_t)
    {
        {
            long l = strtol("0xfffffffff", NULL, 0);
            /* FLAW: Do not check to see if strtol() failed */
            printf("%li\n", l);
        }
    }
    else
    {
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
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* good1() uses if(static_f) instead of if(static_t) */
static void good1()
{
    if(static_f)
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            long l = strtol("0xfffffffff", NULL, 0);
            /* FLAW: Do not check to see if strtol() failed */
            printf("%li\n", l);
        }
    }
    else
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
    }
}

/* good2() reverses the bodies in the if statement */
static void good2()
{
    if(static_t)
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
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            long l = strtol("0xfffffffff", NULL, 0);
            /* FLAW: Do not check to see if strtol() failed */
            printf("%li\n", l);
        }
    }
}

void CWE391_Unchecked_Error_Condition__strtol_05_good()
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
    CWE391_Unchecked_Error_Condition__strtol_05_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE391_Unchecked_Error_Condition__strtol_05_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
