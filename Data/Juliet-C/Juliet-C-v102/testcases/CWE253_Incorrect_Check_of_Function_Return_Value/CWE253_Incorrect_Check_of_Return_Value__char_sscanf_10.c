/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE253_Incorrect_Check_of_Return_Value__char_sscanf_10.c
Label Definition File: CWE253_Incorrect_Check_of_Return_Value.string.label.xml
Template File: point-flaw-10.tmpl.c
*/
/*
 * @description
 * CWE: 253 Incorrect Check of Return Value
 * Sinks: sscanf
 *    GoodSink: Correctly check if sscanf() failed
 *    BadSink : Incorrectly check if sscanf() failed
 * Flow Variant: 10 Control flow: if(global_t) and if(global_f)
 *
 * */

#include "std_testcase.h"

#define SRC "sscanf"

#ifndef OMITBAD

void CWE253_Incorrect_Check_of_Return_Value__char_sscanf_10_bad()
{
    if(global_t)
    {
        {
            /* By initializing data_buf, we ensure this will not be the
             * CWE 690 (Unchecked Return Value To NULL Pointer) flaw for fgets() and other variants */
            char data_buf[100] = "";
            char * data = data_buf;
            /* FLAW: sscanf() might fail, in which case the return value will be EOF (-1), but
             * we are checking to see if the return value is 0 */
            if (sscanf(SRC, "%99s\0", data) == 0)
            {
                printLine("sscanf failed!");
            }
        }
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            /* By initializing data_buf, we ensure this will not be the
             * CWE 690 (Unchecked Return Value To NULL Pointer) flaw for fgets() and other variants */
            char data_buf[100] = "";
            char * data = data_buf;
            /* FIX: check for the correct return value */
            if (sscanf(SRC, "%99s\0", data) == EOF)
            {
                printLine("sscanf failed!");
            }
        }
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* good1() uses if(global_f) instead of if(global_t) */
static void good1()
{
    if(global_f)
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            /* By initializing data_buf, we ensure this will not be the
             * CWE 690 (Unchecked Return Value To NULL Pointer) flaw for fgets() and other variants */
            char data_buf[100] = "";
            char * data = data_buf;
            /* FLAW: sscanf() might fail, in which case the return value will be EOF (-1), but
             * we are checking to see if the return value is 0 */
            if (sscanf(SRC, "%99s\0", data) == 0)
            {
                printLine("sscanf failed!");
            }
        }
    }
    else
    {
        {
            /* By initializing data_buf, we ensure this will not be the
             * CWE 690 (Unchecked Return Value To NULL Pointer) flaw for fgets() and other variants */
            char data_buf[100] = "";
            char * data = data_buf;
            /* FIX: check for the correct return value */
            if (sscanf(SRC, "%99s\0", data) == EOF)
            {
                printLine("sscanf failed!");
            }
        }
    }
}

/* good2() reverses the bodies in the if statement */
static void good2()
{
    if(global_t)
    {
        {
            /* By initializing data_buf, we ensure this will not be the
             * CWE 690 (Unchecked Return Value To NULL Pointer) flaw for fgets() and other variants */
            char data_buf[100] = "";
            char * data = data_buf;
            /* FIX: check for the correct return value */
            if (sscanf(SRC, "%99s\0", data) == EOF)
            {
                printLine("sscanf failed!");
            }
        }
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            /* By initializing data_buf, we ensure this will not be the
             * CWE 690 (Unchecked Return Value To NULL Pointer) flaw for fgets() and other variants */
            char data_buf[100] = "";
            char * data = data_buf;
            /* FLAW: sscanf() might fail, in which case the return value will be EOF (-1), but
             * we are checking to see if the return value is 0 */
            if (sscanf(SRC, "%99s\0", data) == 0)
            {
                printLine("sscanf failed!");
            }
        }
    }
}

void CWE253_Incorrect_Check_of_Return_Value__char_sscanf_10_good()
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
    CWE253_Incorrect_Check_of_Return_Value__char_sscanf_10_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE253_Incorrect_Check_of_Return_Value__char_sscanf_10_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
