/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE252_Unchecked_Return_Value__char_snprintf_18.c
Label Definition File: CWE252_Unchecked_Return_Value.string.label.xml
Template File: point-flaw-18.tmpl.c
*/
/*
 * @description
 * CWE: 252 Unchecked Return Value
 * Sinks: snprintf
 *    GoodSink: Check if snprintf() fails
 *    BadSink : Do not check if snprintf() fails
 * Flow Variant: 18 Control flow: goto statements
 *
 * */

#include "std_testcase.h"

#define SRC "string"

#ifndef OMITBAD

void CWE252_Unchecked_Return_Value__char_snprintf_18_bad()
{
    goto sink;
    /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
    {
        /* By initializing data_buf, we ensure this will not be the
         * CWE 690 (Unchecked Return Value To NULL Pointer) flaw for fgets() and other variants */
        char data_buf[100] = "";
        char * data = data_buf;
        /* FIX: check the return value */
        if (_snprintf(data,100-strlen(SRC)-1, "%s\n", SRC) < 0)
        {
            printLine("snprintf failed!");
        }
    }
sink:
    {
        /* By initializing data_buf, we ensure this will not be the
         * CWE 690 (Unchecked Return Value To NULL Pointer) flaw for fgets() and other variants */
        char data_buf[100] = "";
        char * data = data_buf;
        /* FLAW: Do not check the return value */
        _snprintf(data,100-strlen(SRC)-1, "%s\n", SRC);
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
        /* By initializing data_buf, we ensure this will not be the
         * CWE 690 (Unchecked Return Value To NULL Pointer) flaw for fgets() and other variants */
        char data_buf[100] = "";
        char * data = data_buf;
        /* FLAW: Do not check the return value */
        _snprintf(data,100-strlen(SRC)-1, "%s\n", SRC);
    }
sink:
    {
        /* By initializing data_buf, we ensure this will not be the
         * CWE 690 (Unchecked Return Value To NULL Pointer) flaw for fgets() and other variants */
        char data_buf[100] = "";
        char * data = data_buf;
        /* FIX: check the return value */
        if (_snprintf(data,100-strlen(SRC)-1, "%s\n", SRC) < 0)
        {
            printLine("snprintf failed!");
        }
    }
}

void CWE252_Unchecked_Return_Value__char_snprintf_18_good()
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
    CWE252_Unchecked_Return_Value__char_snprintf_18_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE252_Unchecked_Return_Value__char_snprintf_18_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
