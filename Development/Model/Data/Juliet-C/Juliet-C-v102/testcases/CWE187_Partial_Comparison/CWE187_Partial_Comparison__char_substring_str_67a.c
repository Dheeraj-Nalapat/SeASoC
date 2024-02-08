/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE187_Partial_Comparison__char_substring_str_67a.c
Label Definition File: CWE187_Partial_Comparison.label.xml
Template File: sources-sinks-67a.tmpl.c
*/
/*
 * @description
 * CWE: 187 Partial Comparison
 * BadSource: substring Provide a password that is a shortened version of the actual password
 * GoodSource: Provide a matching password
 * Sinks: str
 *    GoodSink: Compare the 2 passwords correctly
 *    BadSink : use strstr() to do password match, which is a partial comparison
 * Flow Variant: 67 Data flow: data passed in a struct from one function to another in different source files
 *
 * */

#include "std_testcase.h"

#include <wchar.h>

#define PASSWORD "Password1234"
/* PASSWORD_SZ must equal the length of PASSWORD */
#define PASSWORD_SZ strlen(PASSWORD)

typedef struct _CWE187_Partial_Comparison__char_substring_str_67_struct_type
{
    char * a;
} CWE187_Partial_Comparison__char_substring_str_67_struct_type;

#ifndef OMITBAD

/* bad function declaration */
void CWE187_Partial_Comparison__char_substring_str_67b_bad_sink(CWE187_Partial_Comparison__char_substring_str_67_struct_type my_struct);

void CWE187_Partial_Comparison__char_substring_str_67_bad()
{
    char * data;
    CWE187_Partial_Comparison__char_substring_str_67_struct_type my_struct;
    char data_buf[100] = "";
    data = data_buf;
    /* FLAW: Provide a shortened version of the actual password
     * NOTE: This must be a substring of PASSWORD starting with the first character in PASSWORD
     * i.e. other examples could be "Pa", "Pas", "Pass", etc. as long as it is shorter than MIN_PASSWORD_SZ
     * and does not match the full PASSWORD string */
    data = "P";
    my_struct.a = data;
    CWE187_Partial_Comparison__char_substring_str_67b_bad_sink(my_struct);
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B uses the GoodSource with the BadSink */
void CWE187_Partial_Comparison__char_substring_str_67b_goodG2B_sink(CWE187_Partial_Comparison__char_substring_str_67_struct_type my_struct);

static void goodG2B()
{
    char * data;
    CWE187_Partial_Comparison__char_substring_str_67_struct_type my_struct;
    char data_buf[100] = "";
    data = data_buf;
    /* FIX: Use the matching string */
    data = PASSWORD;
    my_struct.a = data;
    CWE187_Partial_Comparison__char_substring_str_67b_goodG2B_sink(my_struct);
}

/* goodB2G uses the BadSource with the GoodSink */
void CWE187_Partial_Comparison__char_substring_str_67b_goodB2G_sink(CWE187_Partial_Comparison__char_substring_str_67_struct_type my_struct);

static void goodB2G()
{
    char * data;
    CWE187_Partial_Comparison__char_substring_str_67_struct_type my_struct;
    char data_buf[100] = "";
    data = data_buf;
    /* FLAW: Provide a shortened version of the actual password
     * NOTE: This must be a substring of PASSWORD starting with the first character in PASSWORD
     * i.e. other examples could be "Pa", "Pas", "Pass", etc. as long as it is shorter than MIN_PASSWORD_SZ
     * and does not match the full PASSWORD string */
    data = "P";
    my_struct.a = data;
    CWE187_Partial_Comparison__char_substring_str_67b_goodB2G_sink(my_struct);
}

void CWE187_Partial_Comparison__char_substring_str_67_good()
{
    goodG2B();
    goodB2G();
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
    CWE187_Partial_Comparison__char_substring_str_67_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE187_Partial_Comparison__char_substring_str_67_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
