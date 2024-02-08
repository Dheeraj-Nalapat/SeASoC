/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE253_Incorrect_Check_of_Return_Value__char_remove_14.c
Label Definition File: CWE253_Incorrect_Check_of_Return_Value.string.label.xml
Template File: point-flaw-14.tmpl.c
*/
/*
 * @description
 * CWE: 253 Incorrect Check of Return Value
 * Sinks: remove
 *    GoodSink: Correctly check if remove() failed
 *    BadSink : Incorrectly check if remove() failed
 * Flow Variant: 14 Control flow: if(global_five==5) and if(global_five!=5)
 *
 * */

#include "std_testcase.h"

#ifdef _WIN32
# define REMOVE remove
#else
# define REMOVE remove
#endif

#ifndef OMITBAD

void CWE253_Incorrect_Check_of_Return_Value__char_remove_14_bad()
{
    if(global_five==5)
    {
        {
            /* FLAW: remove() might fail, in which case the return value will be non-zero, but
             * we are checking to see if the return value is 0 */
            if (REMOVE("removemebad.txt") == 0)
            {
                printLine("remove failed!");
            }
        }
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            /* FIX: check for the correct return value */
            if (REMOVE("removemebad.txt") != 0)
            {
                printLine("remove failed!");
            }
        }
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* good1() uses if(global_five!=5) instead of if(global_five==5) */
static void good1()
{
    if(global_five!=5)
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            /* FLAW: remove() might fail, in which case the return value will be non-zero, but
             * we are checking to see if the return value is 0 */
            if (REMOVE("removemebad.txt") == 0)
            {
                printLine("remove failed!");
            }
        }
    }
    else
    {
        {
            /* FIX: check for the correct return value */
            if (REMOVE("removemebad.txt") != 0)
            {
                printLine("remove failed!");
            }
        }
    }
}

/* good2() reverses the bodies in the if statement */
static void good2()
{
    if(global_five==5)
    {
        {
            /* FIX: check for the correct return value */
            if (REMOVE("removemebad.txt") != 0)
            {
                printLine("remove failed!");
            }
        }
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            /* FLAW: remove() might fail, in which case the return value will be non-zero, but
             * we are checking to see if the return value is 0 */
            if (REMOVE("removemebad.txt") == 0)
            {
                printLine("remove failed!");
            }
        }
    }
}

void CWE253_Incorrect_Check_of_Return_Value__char_remove_14_good()
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
    CWE253_Incorrect_Check_of_Return_Value__char_remove_14_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE253_Incorrect_Check_of_Return_Value__char_remove_14_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
