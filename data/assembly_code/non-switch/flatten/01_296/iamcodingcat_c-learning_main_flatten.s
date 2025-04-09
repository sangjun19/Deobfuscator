	.file	"iamcodingcat_c-learning_main_flatten.c"
	.text
	.globl	_TIG_IZ_4e6E_envp
	.bss
	.align 8
	.type	_TIG_IZ_4e6E_envp, @object
	.size	_TIG_IZ_4e6E_envp, 8
_TIG_IZ_4e6E_envp:
	.zero	8
	.globl	_TIG_IZ_4e6E_argv
	.align 8
	.type	_TIG_IZ_4e6E_argv, @object
	.size	_TIG_IZ_4e6E_argv, 8
_TIG_IZ_4e6E_argv:
	.zero	8
	.globl	_TIG_IZ_4e6E_argc
	.align 4
	.type	_TIG_IZ_4e6E_argc, @object
	.size	_TIG_IZ_4e6E_argc, 4
_TIG_IZ_4e6E_argc:
	.zero	4
	.section	.rodata
.LC0:
	.string	"* "
.LC1:
	.string	"%d"
.LC2:
	.string	"  "
.LC3:
	.string	"Total(while): %d\n"
.LC4:
	.string	"Total(for): %d\n"
	.text
	.globl	main
	.type	main, @function
main:
.LFB3:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$112, %rsp
	movl	%edi, -84(%rbp)
	movq	%rsi, -96(%rbp)
	movq	%rdx, -104(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_4e6E_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_4e6E_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_4e6E_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 138 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-4e6E--0
# 0 "" 2
#NO_APP
	movl	-84(%rbp), %eax
	movl	%eax, _TIG_IZ_4e6E_argc(%rip)
	movq	-96(%rbp), %rax
	movq	%rax, _TIG_IZ_4e6E_argv(%rip)
	movq	-104(%rbp), %rax
	movq	%rax, _TIG_IZ_4e6E_envp(%rip)
	nop
	movq	$77, -16(%rbp)
.L91:
	cmpq	$84, -16(%rbp)
	ja	.L94
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
	.long	.L56-.L8
	.long	.L94-.L8
	.long	.L94-.L8
	.long	.L55-.L8
	.long	.L94-.L8
	.long	.L94-.L8
	.long	.L54-.L8
	.long	.L94-.L8
	.long	.L94-.L8
	.long	.L94-.L8
	.long	.L53-.L8
	.long	.L94-.L8
	.long	.L52-.L8
	.long	.L94-.L8
	.long	.L51-.L8
	.long	.L94-.L8
	.long	.L50-.L8
	.long	.L49-.L8
	.long	.L48-.L8
	.long	.L94-.L8
	.long	.L94-.L8
	.long	.L47-.L8
	.long	.L94-.L8
	.long	.L94-.L8
	.long	.L46-.L8
	.long	.L94-.L8
	.long	.L94-.L8
	.long	.L45-.L8
	.long	.L94-.L8
	.long	.L44-.L8
	.long	.L43-.L8
	.long	.L42-.L8
	.long	.L41-.L8
	.long	.L40-.L8
	.long	.L94-.L8
	.long	.L39-.L8
	.long	.L38-.L8
	.long	.L37-.L8
	.long	.L94-.L8
	.long	.L94-.L8
	.long	.L36-.L8
	.long	.L94-.L8
	.long	.L35-.L8
	.long	.L34-.L8
	.long	.L33-.L8
	.long	.L32-.L8
	.long	.L94-.L8
	.long	.L94-.L8
	.long	.L31-.L8
	.long	.L94-.L8
	.long	.L30-.L8
	.long	.L29-.L8
	.long	.L94-.L8
	.long	.L94-.L8
	.long	.L28-.L8
	.long	.L27-.L8
	.long	.L26-.L8
	.long	.L25-.L8
	.long	.L24-.L8
	.long	.L23-.L8
	.long	.L94-.L8
	.long	.L22-.L8
	.long	.L21-.L8
	.long	.L20-.L8
	.long	.L19-.L8
	.long	.L18-.L8
	.long	.L94-.L8
	.long	.L17-.L8
	.long	.L94-.L8
	.long	.L94-.L8
	.long	.L94-.L8
	.long	.L94-.L8
	.long	.L16-.L8
	.long	.L94-.L8
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L94-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L94-.L8
	.long	.L94-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L48:
	movl	$0, -24(%rbp)
	movq	$17, -16(%rbp)
	jmp	.L57
.L30:
	movl	$0, -32(%rbp)
	movq	$12, -16(%rbp)
	jmp	.L57
.L10:
	cmpl	$9, -64(%rbp)
	jg	.L58
	movq	$51, -16(%rbp)
	jmp	.L57
.L58:
	movq	$6, -16(%rbp)
	jmp	.L57
.L43:
	cmpl	$5, -40(%rbp)
	jg	.L60
	movq	$58, -16(%rbp)
	jmp	.L57
.L60:
	movq	$50, -16(%rbp)
	jmp	.L57
.L21:
	movl	$0, -28(%rbp)
	movq	$43, -16(%rbp)
	jmp	.L57
.L51:
	cmpl	$2, -72(%rbp)
	jg	.L62
	movq	$24, -16(%rbp)
	jmp	.L57
.L62:
	movq	$57, -16(%rbp)
	jmp	.L57
.L26:
	movl	$1, -76(%rbp)
	movq	$21, -16(%rbp)
	jmp	.L57
.L11:
	movl	-24(%rbp), %eax
	addl	$4, %eax
	cmpl	%eax, -20(%rbp)
	jg	.L64
	movq	$67, -16(%rbp)
	jmp	.L57
.L64:
	movq	$37, -16(%rbp)
	jmp	.L57
.L42:
	movl	$0, -20(%rbp)
	movq	$79, -16(%rbp)
	jmp	.L57
.L52:
	cmpl	$4, -32(%rbp)
	jg	.L66
	movq	$62, -16(%rbp)
	jmp	.L57
.L66:
	movq	$18, -16(%rbp)
	jmp	.L57
.L32:
	addl	$1, -20(%rbp)
	movq	$79, -16(%rbp)
	jmp	.L57
.L28:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addl	$1, -44(%rbp)
	movq	$35, -16(%rbp)
	jmp	.L57
.L12:
	movq	$75, -16(%rbp)
	jmp	.L57
.L55:
	movl	$10, %edi
	call	putchar@PLT
	addl	$1, -32(%rbp)
	movq	$12, -16(%rbp)
	jmp	.L57
.L50:
	movl	$10, %edi
	call	putchar@PLT
	addl	$1, -40(%rbp)
	movq	$30, -16(%rbp)
	jmp	.L57
.L46:
	leaq	-76(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$84, -16(%rbp)
	jmp	.L57
.L47:
	movl	$1, -68(%rbp)
	movq	$44, -16(%rbp)
	jmp	.L57
.L38:
	movl	$10, %edi
	call	putchar@PLT
	addl	$1, -72(%rbp)
	movq	$14, -16(%rbp)
	jmp	.L57
.L13:
	movl	-36(%rbp), %eax
	cmpl	-40(%rbp), %eax
	jge	.L68
	movq	$61, -16(%rbp)
	jmp	.L57
.L68:
	movq	$16, -16(%rbp)
	jmp	.L57
.L25:
	movl	$0, -64(%rbp)
	movl	$0, -60(%rbp)
	movq	$80, -16(%rbp)
	jmp	.L57
.L20:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$48, -16(%rbp)
	jmp	.L57
.L29:
	addl	$1, -64(%rbp)
	movl	-64(%rbp), %eax
	addl	%eax, -60(%rbp)
	movq	$80, -16(%rbp)
	jmp	.L57
.L41:
	movl	-32(%rbp), %edx
	movl	-28(%rbp), %eax
	addl	%edx, %eax
	cmpl	$3, %eax
	jg	.L70
	movq	$63, -16(%rbp)
	jmp	.L57
.L70:
	movq	$27, -16(%rbp)
	jmp	.L57
.L49:
	cmpl	$4, -24(%rbp)
	jg	.L72
	movq	$31, -16(%rbp)
	jmp	.L57
.L72:
	movq	$59, -16(%rbp)
	jmp	.L57
.L36:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$45, -16(%rbp)
	jmp	.L57
.L17:
	movl	-24(%rbp), %edx
	movl	-20(%rbp), %eax
	addl	%edx, %eax
	cmpl	$3, %eax
	jg	.L74
	movq	$33, -16(%rbp)
	jmp	.L57
.L74:
	movq	$40, -16(%rbp)
	jmp	.L57
.L27:
	cmpl	$4, -48(%rbp)
	jg	.L76
	movq	$42, -16(%rbp)
	jmp	.L57
.L76:
	movq	$10, -16(%rbp)
	jmp	.L57
.L23:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L92
	jmp	.L93
.L54:
	movl	-60(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	$0, -56(%rbp)
	movl	$1, -52(%rbp)
	movq	$64, -16(%rbp)
	jmp	.L57
.L45:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$48, -16(%rbp)
	jmp	.L57
.L22:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addl	$1, -36(%rbp)
	movq	$76, -16(%rbp)
	jmp	.L57
.L24:
	movl	$0, -36(%rbp)
	movq	$76, -16(%rbp)
	jmp	.L57
.L7:
	movl	-76(%rbp), %eax
	cmpl	$9, %eax
	jle	.L79
	movq	$0, -16(%rbp)
	jmp	.L57
.L79:
	movq	$65, -16(%rbp)
	jmp	.L57
.L15:
	movl	$10, %edi
	call	putchar@PLT
	addl	$1, -48(%rbp)
	movq	$55, -16(%rbp)
	jmp	.L57
.L14:
	movl	$0, -76(%rbp)
	movl	$1, -72(%rbp)
	movq	$14, -16(%rbp)
	jmp	.L57
.L31:
	addl	$1, -28(%rbp)
	movq	$43, -16(%rbp)
	jmp	.L57
.L18:
	movl	-76(%rbp), %eax
	testl	%eax, %eax
	jg	.L81
	movq	$56, -16(%rbp)
	jmp	.L57
.L81:
	movq	$21, -16(%rbp)
	jmp	.L57
.L33:
	movl	-76(%rbp), %eax
	cmpl	%eax, -68(%rbp)
	jg	.L83
	movq	$29, -16(%rbp)
	jmp	.L57
.L83:
	movq	$36, -16(%rbp)
	jmp	.L57
.L16:
	movl	-56(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	$0, -48(%rbp)
	movq	$55, -16(%rbp)
	jmp	.L57
.L40:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$45, -16(%rbp)
	jmp	.L57
.L37:
	movl	$10, %edi
	call	putchar@PLT
	addl	$1, -24(%rbp)
	movq	$17, -16(%rbp)
	jmp	.L57
.L19:
	cmpl	$10, -52(%rbp)
	jg	.L85
	movq	$83, -16(%rbp)
	jmp	.L57
.L85:
	movq	$72, -16(%rbp)
	jmp	.L57
.L53:
	movl	$1, -40(%rbp)
	movq	$30, -16(%rbp)
	jmp	.L57
.L35:
	movl	$0, -44(%rbp)
	movq	$35, -16(%rbp)
	jmp	.L57
.L56:
	movl	$9, -76(%rbp)
	movq	$21, -16(%rbp)
	jmp	.L57
.L9:
	movl	-52(%rbp), %eax
	addl	%eax, -56(%rbp)
	addl	$1, -52(%rbp)
	movq	$64, -16(%rbp)
	jmp	.L57
.L39:
	cmpl	$4, -44(%rbp)
	jg	.L87
	movq	$54, -16(%rbp)
	jmp	.L57
.L87:
	movq	$74, -16(%rbp)
	jmp	.L57
.L44:
	movl	$42, %edi
	call	putchar@PLT
	addl	$1, -68(%rbp)
	movq	$44, -16(%rbp)
	jmp	.L57
.L34:
	cmpl	$4, -28(%rbp)
	jg	.L89
	movq	$32, -16(%rbp)
	jmp	.L57
.L89:
	movq	$3, -16(%rbp)
	jmp	.L57
.L94:
	nop
.L57:
	jmp	.L91
.L93:
	call	__stack_chk_fail@PLT
.L92:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
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
