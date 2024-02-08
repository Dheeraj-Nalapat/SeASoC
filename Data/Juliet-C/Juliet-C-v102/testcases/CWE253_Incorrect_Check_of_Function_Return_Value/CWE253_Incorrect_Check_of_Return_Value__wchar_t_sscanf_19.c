/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE253_Incorrect_Check_of_Return_Value__wchar_t_sscanf_19.c
Label Definition File: CWE253_Incorrect_Check_of_Return_Value.string.label.xml
Template File: point-flaw-19.tmpl.c
*/
/*
 * @description
 * CWE: 253 Incorrect Check of Return Value
 * Sinks: sscanf
 *    GoodSink: Correctly check if swscanf() failed
 *    BadSink : Incorrectly check if swscanf() failed
 * Flow Variant: 19 Control flow: Dead code after a return
 *
 * */

#include "std_testcase.h"

#define SRC L"sscanf"

#ifndef OMITBAD

void CWE253_Incorrect_Check_of_Return_Value__wchar_t_sscanf_19_bad()
{
    {
        /* By initializing data_buf, we ensure this will not be the
         * CWE 690 (Unchecked Return Value To NULL Pointer) flaw for fgetws() and other variants */
        wchar_t data_buf[100] = L"";
        wchar_t * data = data_buf;
        /* FLAW: swscanf() might fail, in which case the return value will be EOF (-1), but
         * we are checking to see if the return value is 0 */
        if (swscanf(SRC, L"%99s\0", data) == 0)
        {
            printLine("swscanf failed!");
        }
    }
    return;
    /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
    {
        /* By initializing data_buf, we ensure this will not be the
         * CWE 690 (Unchecked Return Value To NULL Pointer) flaw for fgetws() and other variants */
        wchar_t data_buf[100] = L"";
        wchar_t * data = data_buf;
        /* FIX: check for the correct return value */
        if (swscanf(SRC, L"%99s\0", data) == EOF)
        {
            printLine("swscanf failed!");
        }
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* good1() reverses the GoodSinkBody and the BadSinkBody so that the BadSinkBody never runs */
static void good1()
{
    {
        /* By initializing data_buf, we ensure this will not be the
         * CWE 690 (Unchecked Return Value To NULL Pointer) flaw for fgetws() and other variants */
        wchar_t data_buf[100] = L"";
        wchar_t * data = data_buf;
        /* FIX: check for the correct return value */
        if (swscanf(SRC, L"%99s\0", data) == EOF)
        {
            printLine("swscanf failed!");
        }
    }
    return;
    /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
    {
        /* By initializing data_buf, we ensure this will not be the
         * CWE 690 (Unchecked Return Value To NULL Pointer) flaw for fgetws() and other variants */
        wchar_t data_buf[100] = L"";
        wchar_t * data = data_buf;
        /* FLAW: swscanf() might fail, in which case the return value will be EOF (-1), but
         * we are checking to see if the return value is 0 */
        if (swscanf(SRC, L"%99s\0", data) == 0)
        {
            printLine("swscanf failed!");
        }
    }
}

void CWE253_Incorrect_Check_of_Return_Value__wchar_t_sscanf_19_good()
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
    CWE253_Incorrect_Check_of_Return_Value__wchar_t_sscanf_19_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE253_Incorrect_Check_of_Return_Value__wchar_t_sscanf_19_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
