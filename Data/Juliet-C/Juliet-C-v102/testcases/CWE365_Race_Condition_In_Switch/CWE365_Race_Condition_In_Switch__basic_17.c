/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE365_Race_Condition_In_Switch__basic_17.c
Label Definition File: CWE365_Race_Condition_In_Switch__basic.label.xml
Template File: point-flaw-17.tmpl.c
*/
/*
 * @description
 * CWE: 365 Race Condition in Switch
 * Sinks:
 *    GoodSink: Switch based on call to time() using a temporary variable
 *    BadSink : Switch based on call to time()
 * Flow Variant: 17 Control flow: for loops
 *
 * */

#include "std_testcase.h"

#include <time.h>

#ifndef OMITBAD

void CWE365_Race_Condition_In_Switch__basic_17_bad()
{
    int j,k;
    for(j = 0; j < 1; j++)
    {
        /* FLAW: switch-ing based on a call to 'time(NULL)' which could change
         * from one call to the next (when the clock is near a second-boundary)
         */
        switch (time(NULL) % 4)
        {
        case 0:
            printLine("0");
            break;
        case 1:
            printLine("1");
            break;
        default:
            printLine("default");
            break;
        }
    }
    for(k = 0; k < 0; k++)
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            time_t time_val = time(NULL);
            /* FIX: assign to a temporary prior to calling */
            switch (time_val % 4)
            {
            case 0:
                printLine("0");
                break;
            case 1:
                printLine("1");
                break;
            default:
                printLine("default");
                break;
            }
        }
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* good1() changes the conditions on the for statements */
static void good1()
{
    int j,k;
    for(j = 0; j < 0; j++)
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* FLAW: switch-ing based on a call to 'time(NULL)' which could change
         * from one call to the next (when the clock is near a second-boundary)
         */
        switch (time(NULL) % 4)
        {
        case 0:
            printLine("0");
            break;
        case 1:
            printLine("1");
            break;
        default:
            printLine("default");
            break;
        }
    }
    for(k = 0; k < 1; k++)
    {
        {
            time_t time_val = time(NULL);
            /* FIX: assign to a temporary prior to calling */
            switch (time_val % 4)
            {
            case 0:
                printLine("0");
                break;
            case 1:
                printLine("1");
                break;
            default:
                printLine("default");
                break;
            }
        }
    }
}

void CWE365_Race_Condition_In_Switch__basic_17_good()
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
    CWE365_Race_Condition_In_Switch__basic_17_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE365_Race_Condition_In_Switch__basic_17_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
