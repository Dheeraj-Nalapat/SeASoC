/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE606_Unchecked_Loop_Condition__char_Environment_54e.c
Label Definition File: CWE606_Unchecked_Loop_Condition.label.xml
Template File: sources-sinks-54e.tmpl.c
*/
/*
 * @description
 * CWE: 606 Unchecked Input For Loop Condition
 * BadSource: Environment Read input from an environment variable
 * GoodSource: Input a number less than MAX_LOOP
 * Sinks:
 *    GoodSink: Use data as the for loop variant after checking to see if it is less than MAX_LOOP
 *    BadSink : Use data as the for loop variant without checking its size
 * Flow Variant: 54 Data flow: data passed as an argument from one function through three others to a fifth; all five functions are in different source files
 *
 * */

#include "std_testcase.h"

#define MAX_LOOP 10000

#define ENV_VARIABLE "ADD"

#ifdef _WIN32
# define GETENV getenv
#else
# define GETENV getenv
#endif

#ifndef OMITBAD

void CWE606_Unchecked_Loop_Condition__char_Environment_54e_bad_sink(char * data)
{
    {
        int i, n, v;
        if (sscanf(data, "%d", &n) == 1)
        {
            /* POTENTIAL FLAW: user-supplied value 'n' could lead to very large loop iteration */
            v = 0;
            for (i = 0; i < n; i++)
            {
                /* INCIDENTAL: CWE 561: Dead Code - non-avoidable if n <= 0 */
                v++; /* avoid a dead/empty code block issue */
            }
            printIntLine(v);
        }
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B uses the GoodSource with the BadSink */
void CWE606_Unchecked_Loop_Condition__char_Environment_54e_goodG2B_sink(char * data)
{
    {
        int i, n, v;
        if (sscanf(data, "%d", &n) == 1)
        {
            /* POTENTIAL FLAW: user-supplied value 'n' could lead to very large loop iteration */
            v = 0;
            for (i = 0; i < n; i++)
            {
                /* INCIDENTAL: CWE 561: Dead Code - non-avoidable if n <= 0 */
                v++; /* avoid a dead/empty code block issue */
            }
            printIntLine(v);
        }
    }
}

/* goodB2G uses the BadSource with the GoodSink */
void CWE606_Unchecked_Loop_Condition__char_Environment_54e_goodB2G_sink(char * data)
{
    {
        int i, n, v;
        if (sscanf(data, "%d", &n) == 1)
        {
            /* FIX: limit loop iteration counts */
            if (n < MAX_LOOP)
            {
                v = 0;
                for (i = 0; i < n; i++)
                {
                    /* INCIDENTAL: CWE 561: Dead Code - non-avoidable if n <= 0 */
                    v++; /* avoid a dead/empty code block issue */
                }
                printIntLine(v);
            }
        }
    }
}

#endif /* OMITGOOD */
