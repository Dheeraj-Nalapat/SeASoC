/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE478_Failure_To_Use_Default_Case_In_Switch__basic_02.c
Label Definition File: CWE478_Failure_To_Use_Default_Case_In_Switch__basic.label.xml
Template File: point-flaw-02.tmpl.c
*/
/*
 * @description
 * CWE: 478 Failure to Use Default Case in Switch
 * Sinks:
 *    GoodSink: Use default case in switch statement
 *    BadSink : No default case in a switch statement
 * Flow Variant: 02 Control flow: if(1) and if(0)
 *
 * */

#include "std_testcase.h"

#ifndef OMITBAD

void CWE478_Failure_To_Use_Default_Case_In_Switch__basic_02_bad()
{
    if(1)
    {
        {
            const char *s = "shouldn\'t see this value";
            int x;
            x = (rand() % 3);
            switch (x)
            {
            case 0:
                s = "0";
                break;
            case 1:
                s = "1";
                break;
                /* FLAW: x could be 2, and there is no 'default' case for that */
            }
            printLine(s);
        }
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            const char *s = "shouldn\'t see this value";
            int x;
            x = (rand() % 3);
            switch (x)
            {
            case 0:
                s = "0";
                break;
            case 1:
                s = "1";
                break;
                /* FIX: provide a 'default' case for other values */
            default:
                return;
            }
            printLine(s);
        }
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* good1() uses if(0) instead of if(1) */
static void good1()
{
    if(0)
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            const char *s = "shouldn\'t see this value";
            int x;
            x = (rand() % 3);
            switch (x)
            {
            case 0:
                s = "0";
                break;
            case 1:
                s = "1";
                break;
                /* FLAW: x could be 2, and there is no 'default' case for that */
            }
            printLine(s);
        }
    }
    else
    {
        {
            const char *s = "shouldn\'t see this value";
            int x;
            x = (rand() % 3);
            switch (x)
            {
            case 0:
                s = "0";
                break;
            case 1:
                s = "1";
                break;
                /* FIX: provide a 'default' case for other values */
            default:
                return;
            }
            printLine(s);
        }
    }
}

/* good2() reverses the bodies in the if statement */
static void good2()
{
    if(1)
    {
        {
            const char *s = "shouldn\'t see this value";
            int x;
            x = (rand() % 3);
            switch (x)
            {
            case 0:
                s = "0";
                break;
            case 1:
                s = "1";
                break;
                /* FIX: provide a 'default' case for other values */
            default:
                return;
            }
            printLine(s);
        }
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            const char *s = "shouldn\'t see this value";
            int x;
            x = (rand() % 3);
            switch (x)
            {
            case 0:
                s = "0";
                break;
            case 1:
                s = "1";
                break;
                /* FLAW: x could be 2, and there is no 'default' case for that */
            }
            printLine(s);
        }
    }
}

void CWE478_Failure_To_Use_Default_Case_In_Switch__basic_02_good()
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
    CWE478_Failure_To_Use_Default_Case_In_Switch__basic_02_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE478_Failure_To_Use_Default_Case_In_Switch__basic_02_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif