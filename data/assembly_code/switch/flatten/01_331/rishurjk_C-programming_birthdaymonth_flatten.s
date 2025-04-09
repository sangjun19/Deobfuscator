	.file	"rishurjk_C-programming_birthdaymonth_flatten.c"
	.text
	.globl	_TIG_IZ_cGFD_envp
	.bss
	.align 8
	.type	_TIG_IZ_cGFD_envp, @object
	.size	_TIG_IZ_cGFD_envp, 8
_TIG_IZ_cGFD_envp:
	.zero	8
	.globl	_TIG_IZ_cGFD_argc
	.align 4
	.type	_TIG_IZ_cGFD_argc, @object
	.size	_TIG_IZ_cGFD_argc, 4
_TIG_IZ_cGFD_argc:
	.zero	4
	.globl	_TIG_IZ_cGFD_argv
	.align 8
	.type	_TIG_IZ_cGFD_argv, @object
	.size	_TIG_IZ_cGFD_argv, 8
_TIG_IZ_cGFD_argv:
	.zero	8
	.section	.rodata
.LC0:
	.string	"June"
.LC1:
	.string	"August"
.LC2:
	.string	"May"
.LC3:
	.string	"September"
.LC4:
	.string	"2nd month is 'February'"
.LC5:
	.string	"1st month is 'January'"
.LC6:
	.string	"March"
.LC7:
	.string	"November"
	.align 8
.LC8:
	.string	"Enter Your Birth Month(in num): "
.LC9:
	.string	"%d"
.LC10:
	.string	"October"
.LC11:
	.string	"April"
.LC12:
	.string	"December"
.LC13:
	.string	"July"
	.text
	.globl	main
	.type	main, @function
main:
.LFB6:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$64, %rsp
	movl	%edi, -36(%rbp)
	movq	%rsi, -48(%rbp)
	movq	%rdx, -56(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_cGFD_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_cGFD_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_cGFD_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 110 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-cGFD--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_cGFD_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_cGFD_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_cGFD_envp(%rip)
	nop
	movq	$17, -16(%rbp)
.L42:
	cmpq	$28, -16(%rbp)
	ja	.L45
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L8(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L8(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L8:
	.long	.L45-.L8
	.long	.L45-.L8
	.long	.L24-.L8
	.long	.L23-.L8
	.long	.L45-.L8
	.long	.L22-.L8
	.long	.L45-.L8
	.long	.L21-.L8
	.long	.L20-.L8
	.long	.L45-.L8
	.long	.L45-.L8
	.long	.L45-.L8
	.long	.L19-.L8
	.long	.L18-.L8
	.long	.L17-.L8
	.long	.L45-.L8
	.long	.L45-.L8
	.long	.L16-.L8
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L45-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L45-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L45-.L8
	.long	.L7-.L8
	.text
.L15:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$25, -16(%rbp)
	jmp	.L25
.L10:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L43
	jmp	.L44
.L17:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$25, -16(%rbp)
	jmp	.L25
.L19:
	movl	-20(%rbp), %eax
	cmpl	$12, %eax
	ja	.L27
	movl	%eax, %eax
	leaq	0(,%rax,4), %rdx
	leaq	.L29(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L29(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L29:
	.long	.L27-.L29
	.long	.L40-.L29
	.long	.L39-.L29
	.long	.L38-.L29
	.long	.L37-.L29
	.long	.L36-.L29
	.long	.L35-.L29
	.long	.L34-.L29
	.long	.L33-.L29
	.long	.L32-.L29
	.long	.L31-.L29
	.long	.L30-.L29
	.long	.L28-.L29
	.text
.L28:
	movq	$5, -16(%rbp)
	jmp	.L41
.L30:
	movq	$13, -16(%rbp)
	jmp	.L41
.L31:
	movq	$22, -16(%rbp)
	jmp	.L41
.L32:
	movq	$3, -16(%rbp)
	jmp	.L41
.L33:
	movq	$14, -16(%rbp)
	jmp	.L41
.L34:
	movq	$2, -16(%rbp)
	jmp	.L41
.L35:
	movq	$18, -16(%rbp)
	jmp	.L41
.L36:
	movq	$8, -16(%rbp)
	jmp	.L41
.L37:
	movq	$28, -16(%rbp)
	jmp	.L41
.L38:
	movq	$26, -16(%rbp)
	jmp	.L41
.L39:
	movq	$24, -16(%rbp)
	jmp	.L41
.L40:
	movq	$21, -16(%rbp)
	jmp	.L41
.L27:
	movq	$7, -16(%rbp)
	nop
.L41:
	jmp	.L25
.L20:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$25, -16(%rbp)
	jmp	.L25
.L23:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$25, -16(%rbp)
	jmp	.L25
.L11:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$25, -16(%rbp)
	jmp	.L25
.L13:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$25, -16(%rbp)
	jmp	.L25
.L9:
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$25, -16(%rbp)
	jmp	.L25
.L18:
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$25, -16(%rbp)
	jmp	.L25
.L14:
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-20(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$12, -16(%rbp)
	jmp	.L25
.L16:
	movq	$19, -16(%rbp)
	jmp	.L25
.L12:
	leaq	.LC10(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$25, -16(%rbp)
	jmp	.L25
.L7:
	leaq	.LC11(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$25, -16(%rbp)
	jmp	.L25
.L22:
	leaq	.LC12(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$25, -16(%rbp)
	jmp	.L25
.L21:
	movq	$25, -16(%rbp)
	jmp	.L25
.L24:
	leaq	.LC13(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$25, -16(%rbp)
	jmp	.L25
.L45:
	nop
.L25:
	jmp	.L42
.L44:
	call	__stack_chk_fail@PLT
.L43:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	main, .-main
	.ident	"GCC: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0"
	.section	.note.GNU-stack,"",@progbits
	.section	.note.gnu.property,"a"
	.align 8
	.long	1f - 0f
	.long	4f - 1f
	.long	5
0:
	.string	"GNU"
1:
	.align 8
	.long	0xc0000002
	.long	3f - 2f
2:
	.long	0x3
3:
	.align 8
4:
