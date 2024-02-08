/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE253_Incorrect_Check_of_Return_Value__char_fgets_15.c
Label Definition File: CWE253_Incorrect_Check_of_Return_Value.string.label.xml
Template File: point-flaw-15.tmpl.c
*/
/*
 * @description
 * CWE: 253 Incorrect Check of Return Value
 * Sinks: fgets
 *    GoodSink: Correctly check if fgets() failed before trying to use the destination buffer
 *    BadSink : Incorrectly check if fgets() failed
 * Flow Variant: 15 Control flow: switch(6)
 *
 * */

#include "std_testcase.h"

#ifndef OMITBAD

void CWE253_Incorrect_Check_of_Return_Value__char_fgets_15_bad()
{
    switch(6)
    {
    case 6:
    {
        /* By initializing data_buf, we ensure this will not be the
         * CWE 690 (Unchecked Return Value To NULL Pointer) flaw for fgets() and other variants */
        char data_buf[100] = "";
        char * data = data_buf;
        printLine("Please enter a string: ");
        /* FLAW: Incorrectly check the return value: if this function failed it will return NULL,
         * but we are checking to see if the return value is less than 0. */
        if (fgets(data, 100, stdin) < 0)
        {
            printLine("fgets failed!");
            exit(1);
        }
        printLine(data);
    }
    break;
    default:
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
    {
        /* By initializing data_buf, we ensure this will not be the
         * CWE 690 (Unchecked Return Value To NULL Pointer) flaw for fgets() and other variants */
        char data_buf[100] = "";
        char * data = data_buf;
        printLine("Please enter a string: ");
        /* FIX: check for the correct return value */
        if (fgets(data, 100, stdin) == NULL)
        {
            printLine("fgets failed!");
            exit(1);
        }
        printLine(data);
    }
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* good1() changes the switch to switch(5) */
static void good1()
{
    switch(5)
    {
    case 6:
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
    {
        /* By initializing data_buf, we ensure this will not be the
         * CWE 690 (Unchecked Return Value To NULL Pointer) flaw for fgets() and other variants */
        char data_buf[100] = "";
        char * data = data_buf;
        printLine("Please enter a string: ");
        /* FLAW: Incorrectly check the return value: if this function failed it will return NULL,
         * but we are checking to see if the return value is less than 0. */
        if (fgets(data, 100, stdin) < 0)
        {
            printLine("fgets failed!");
            exit(1);
        }
        printLine(data);
    }
    break;
    default:
    {
        /* By initializing data_buf, we ensure this will not be the
         * CWE 690 (Unchecked Return Value To NULL Pointer) flaw for fgets() and other variants */
        char data_buf[100] = "";
        char * data = data_buf;
        printLine("Please enter a string: ");
        /* FIX: check for the correct return value */
        if (fgets(data, 100, stdin) == NULL)
        {
            printLine("fgets failed!");
            exit(1);
        }
        printLine(data);
    }
    }
}

/* good2() reverses the blocks in the switch */
static void good2()
{
    switch(6)
    {
    case 6:
    {
        /* By initializing data_buf, we ensure this will not be the
         * CWE 690 (Unchecked Return Value To NULL Pointer) flaw for fgets() and other variants */
        char data_buf[100] = "";
        char * data = data_buf;
        printLine("Please enter a string: ");
        /* FIX: check for the correct return value */
        if (fgets(data, 100, stdin) == NULL)
        {
            printLine("fgets failed!");
            exit(1);
        }
        printLine(data);
    }
    break;
    default:
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
    {
        /* By initializing data_buf, we ensure this will not be the
         * CWE 690 (Unchecked Return Value To NULL Pointer) flaw for fgets() and other variants */
        char data_buf[100] = "";
        char * data = data_buf;
        printLine("Please enter a string: ");
        /* FLAW: Incorrectly check the return value: if this function failed it will return NULL,
         * but we are checking to see if the return value is less than 0. */
        if (fgets(data, 100, stdin) < 0)
        {
            printLine("fgets failed!");
            exit(1);
        }
        printLine(data);
    }
    }
}

void CWE253_Incorrect_Check_of_Return_Value__char_fgets_15_good()
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
    CWE253_Incorrect_Check_of_Return_Value__char_fgets_15_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE253_Incorrect_Check_of_Return_Value__char_fgets_15_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif