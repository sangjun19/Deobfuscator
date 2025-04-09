	.file	"HyperSoWeak_JudgeGirl_50043_flatten.c"
	.text
	.globl	_TIG_IZ_h95f_argv
	.bss
	.align 8
	.type	_TIG_IZ_h95f_argv, @object
	.size	_TIG_IZ_h95f_argv, 8
_TIG_IZ_h95f_argv:
	.zero	8
	.globl	_TIG_IZ_h95f_envp
	.align 8
	.type	_TIG_IZ_h95f_envp, @object
	.size	_TIG_IZ_h95f_envp, 8
_TIG_IZ_h95f_envp:
	.zero	8
	.globl	_TIG_IZ_h95f_argc
	.align 4
	.type	_TIG_IZ_h95f_argc, @object
	.size	_TIG_IZ_h95f_argc, 4
_TIG_IZ_h95f_argc:
	.zero	4
	.section	.rodata
.LC0:
	.string	"%d"
.LC1:
	.string	"%d%d%d%d%d%d"
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
	subq	$224, %rsp
	movl	%edi, -196(%rbp)
	movq	%rsi, -208(%rbp)
	movq	%rdx, -216(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_h95f_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_h95f_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_h95f_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 120 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-h95f--0
# 0 "" 2
#NO_APP
	movl	-196(%rbp), %eax
	movl	%eax, _TIG_IZ_h95f_argc(%rip)
	movq	-208(%rbp), %rax
	movq	%rax, _TIG_IZ_h95f_argv(%rip)
	movq	-216(%rbp), %rax
	movq	%rax, _TIG_IZ_h95f_envp(%rip)
	nop
	movq	$58, -56(%rbp)
.L116:
	cmpq	$73, -56(%rbp)
	ja	.L119
	movq	-56(%rbp), %rax
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
	.long	.L68-.L8
	.long	.L119-.L8
	.long	.L67-.L8
	.long	.L66-.L8
	.long	.L65-.L8
	.long	.L64-.L8
	.long	.L119-.L8
	.long	.L63-.L8
	.long	.L62-.L8
	.long	.L119-.L8
	.long	.L119-.L8
	.long	.L87-.L8
	.long	.L60-.L8
	.long	.L59-.L8
	.long	.L58-.L8
	.long	.L57-.L8
	.long	.L56-.L8
	.long	.L55-.L8
	.long	.L54-.L8
	.long	.L53-.L8
	.long	.L119-.L8
	.long	.L52-.L8
	.long	.L51-.L8
	.long	.L50-.L8
	.long	.L49-.L8
	.long	.L48-.L8
	.long	.L119-.L8
	.long	.L47-.L8
	.long	.L119-.L8
	.long	.L46-.L8
	.long	.L45-.L8
	.long	.L119-.L8
	.long	.L44-.L8
	.long	.L43-.L8
	.long	.L42-.L8
	.long	.L41-.L8
	.long	.L119-.L8
	.long	.L40-.L8
	.long	.L39-.L8
	.long	.L38-.L8
	.long	.L37-.L8
	.long	.L119-.L8
	.long	.L36-.L8
	.long	.L35-.L8
	.long	.L34-.L8
	.long	.L33-.L8
	.long	.L32-.L8
	.long	.L31-.L8
	.long	.L30-.L8
	.long	.L119-.L8
	.long	.L29-.L8
	.long	.L28-.L8
	.long	.L27-.L8
	.long	.L26-.L8
	.long	.L25-.L8
	.long	.L24-.L8
	.long	.L23-.L8
	.long	.L22-.L8
	.long	.L21-.L8
	.long	.L20-.L8
	.long	.L19-.L8
	.long	.L18-.L8
	.long	.L17-.L8
	.long	.L16-.L8
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L119-.L8
	.long	.L13-.L8
	.long	.L119-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L54:
	movl	-80(%rbp), %eax
	cltq
	movl	-48(%rbp,%rax,4), %edx
	movl	-88(%rbp), %eax
	addl	%edx, %eax
	movl	%eax, -64(%rbp)
	movl	-80(%rbp), %eax
	cltq
	movl	-32(%rbp,%rax,4), %edx
	movl	-84(%rbp), %eax
	addl	%edx, %eax
	movl	%eax, -60(%rbp)
	movl	-180(%rbp), %eax
	subl	-64(%rbp), %eax
	movl	%eax, %edx
	movl	-168(%rbp), %eax
	subl	-60(%rbp), %eax
	imull	%eax, %edx
	movl	-172(%rbp), %eax
	subl	-64(%rbp), %eax
	movl	%eax, %ecx
	movl	-176(%rbp), %eax
	subl	-60(%rbp), %eax
	imull	%eax, %ecx
	movl	%edx, %eax
	subl	%ecx, %eax
	movl	%eax, -76(%rbp)
	movl	-172(%rbp), %eax
	subl	-64(%rbp), %eax
	movl	%eax, %edx
	movl	-160(%rbp), %eax
	subl	-60(%rbp), %eax
	imull	%eax, %edx
	movl	-164(%rbp), %eax
	subl	-64(%rbp), %eax
	movl	%eax, %ecx
	movl	-168(%rbp), %eax
	subl	-60(%rbp), %eax
	imull	%eax, %ecx
	movl	%edx, %eax
	subl	%ecx, %eax
	movl	%eax, -72(%rbp)
	movl	-164(%rbp), %eax
	subl	-64(%rbp), %eax
	movl	%eax, %edx
	movl	-176(%rbp), %eax
	subl	-60(%rbp), %eax
	imull	%eax, %edx
	movl	-180(%rbp), %eax
	subl	-64(%rbp), %eax
	movl	%eax, %ecx
	movl	-160(%rbp), %eax
	subl	-60(%rbp), %eax
	imull	%eax, %ecx
	movl	%edx, %eax
	subl	%ecx, %eax
	movl	%eax, -68(%rbp)
	movq	$51, -56(%rbp)
	jmp	.L69
.L29:
	movl	-164(%rbp), %eax
	movl	%eax, -140(%rbp)
	movq	$13, -56(%rbp)
	jmp	.L69
.L48:
	addl	$1, -80(%rbp)
	movq	$57, -56(%rbp)
	jmp	.L69
.L27:
	cmpl	$0, -68(%rbp)
	jg	.L70
	movq	$29, -56(%rbp)
	jmp	.L69
.L70:
	movq	$39, -56(%rbp)
	jmp	.L69
.L65:
	movl	-132(%rbp), %eax
	movl	%eax, -128(%rbp)
	movq	$48, -56(%rbp)
	jmp	.L69
.L45:
	movl	-164(%rbp), %eax
	movl	%eax, -116(%rbp)
	movq	$54, -56(%rbp)
	jmp	.L69
.L17:
	movl	-176(%rbp), %eax
	cmpl	%eax, -92(%rbp)
	jle	.L72
	movq	$8, -56(%rbp)
	jmp	.L69
.L72:
	movq	$46, -56(%rbp)
	jmp	.L69
.L58:
	movl	-176(%rbp), %eax
	movl	%eax, -128(%rbp)
	movq	$48, -56(%rbp)
	jmp	.L69
.L57:
	cmpl	$0, -68(%rbp)
	js	.L74
	movq	$29, -56(%rbp)
	jmp	.L69
.L74:
	movq	$11, -56(%rbp)
	jmp	.L69
.L23:
	movl	$0, -80(%rbp)
	movq	$57, -56(%rbp)
	jmp	.L69
.L60:
	movl	-100(%rbp), %eax
	movl	%eax, -96(%rbp)
	movq	$7, -56(%rbp)
	jmp	.L69
.L12:
	movl	-172(%rbp), %eax
	movl	%eax, -108(%rbp)
	movq	$44, -56(%rbp)
	jmp	.L69
.L62:
	movl	-176(%rbp), %eax
	movl	%eax, -96(%rbp)
	movq	$7, -56(%rbp)
	jmp	.L69
.L33:
	movl	-88(%rbp), %eax
	cmpl	-152(%rbp), %eax
	jg	.L76
	movq	$64, -56(%rbp)
	jmp	.L69
.L76:
	movq	$3, -56(%rbp)
	jmp	.L69
.L25:
	movl	-116(%rbp), %eax
	movl	%eax, -112(%rbp)
	movq	$59, -56(%rbp)
	jmp	.L69
.L50:
	movl	-172(%rbp), %edx
	movl	-164(%rbp), %eax
	cmpl	%eax, %edx
	jle	.L78
	movq	$73, -56(%rbp)
	jmp	.L69
.L78:
	movq	$0, -56(%rbp)
	jmp	.L69
.L11:
	cmpl	$0, -72(%rbp)
	js	.L80
	movq	$15, -56(%rbp)
	jmp	.L69
.L80:
	movq	$11, -56(%rbp)
	jmp	.L69
.L66:
	movl	-156(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$16, -56(%rbp)
	jmp	.L69
.L56:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L117
	jmp	.L118
.L49:
	movl	-172(%rbp), %edx
	movl	-164(%rbp), %eax
	cmpl	%eax, %edx
	jge	.L83
	movq	$5, -56(%rbp)
	jmp	.L69
.L83:
	movq	$30, -56(%rbp)
	jmp	.L69
.L52:
	movl	$0, -156(%rbp)
	movl	$0, -48(%rbp)
	movl	$1, -44(%rbp)
	movl	$0, -40(%rbp)
	movl	$1, -36(%rbp)
	movl	$0, -32(%rbp)
	movl	$0, -28(%rbp)
	movl	$-1, -24(%rbp)
	movl	$-1, -20(%rbp)
	leaq	-164(%rbp), %r8
	leaq	-168(%rbp), %rdi
	leaq	-172(%rbp), %rcx
	leaq	-176(%rbp), %rdx
	leaq	-180(%rbp), %rax
	subq	$8, %rsp
	leaq	-160(%rbp), %rsi
	pushq	%rsi
	movq	%r8, %r9
	movq	%rdi, %r8
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	addq	$16, %rsp
	movq	$67, -56(%rbp)
	jmp	.L69
.L22:
	cmpl	$3, -80(%rbp)
	jg	.L85
	movq	$18, -56(%rbp)
	jmp	.L69
.L85:
	movq	$39, -56(%rbp)
	jmp	.L69
.L61:
.L87:
	cmpl	$0, -76(%rbp)
	jg	.L88
	movq	$60, -56(%rbp)
	jmp	.L69
.L88:
	movq	$39, -56(%rbp)
	jmp	.L69
.L59:
	movl	-180(%rbp), %eax
	cmpl	%eax, -140(%rbp)
	jge	.L90
	movq	$17, -56(%rbp)
	jmp	.L69
.L90:
	movq	$23, -56(%rbp)
	jmp	.L69
.L16:
	movl	-168(%rbp), %eax
	movl	%eax, -132(%rbp)
	movq	$4, -56(%rbp)
	jmp	.L69
.L28:
	cmpl	$0, -76(%rbp)
	js	.L92
	movq	$70, -56(%rbp)
	jmp	.L69
.L92:
	movq	$11, -56(%rbp)
	jmp	.L69
.L53:
	movl	-168(%rbp), %edx
	movl	-160(%rbp), %eax
	cmpl	%eax, %edx
	jle	.L94
	movq	$40, -56(%rbp)
	jmp	.L69
.L94:
	movq	$37, -56(%rbp)
	jmp	.L69
.L44:
	movl	-160(%rbp), %eax
	movl	%eax, -132(%rbp)
	movq	$4, -56(%rbp)
	jmp	.L69
.L55:
	movl	-180(%rbp), %eax
	movl	%eax, -144(%rbp)
	movq	$47, -56(%rbp)
	jmp	.L69
.L37:
	movl	-168(%rbp), %eax
	movl	%eax, -124(%rbp)
	movq	$22, -56(%rbp)
	jmp	.L69
.L13:
	movl	-172(%rbp), %edx
	movl	-164(%rbp), %eax
	cmpl	%eax, %edx
	jle	.L96
	movq	$65, -56(%rbp)
	jmp	.L69
.L96:
	movq	$50, -56(%rbp)
	jmp	.L69
.L24:
	movl	-148(%rbp), %eax
	movl	%eax, -144(%rbp)
	movq	$47, -56(%rbp)
	jmp	.L69
.L19:
	cmpl	$0, -72(%rbp)
	jg	.L98
	movq	$52, -56(%rbp)
	jmp	.L69
.L98:
	movq	$39, -56(%rbp)
	jmp	.L69
.L20:
	movl	-112(%rbp), %eax
	movl	%eax, -120(%rbp)
	movq	$38, -56(%rbp)
	jmp	.L69
.L47:
	movl	-168(%rbp), %edx
	movl	-160(%rbp), %eax
	cmpl	%eax, %edx
	jle	.L100
	movq	$63, -56(%rbp)
	jmp	.L69
.L100:
	movq	$32, -56(%rbp)
	jmp	.L69
.L39:
	movl	-168(%rbp), %edx
	movl	-160(%rbp), %eax
	cmpl	%eax, %edx
	jge	.L102
	movq	$72, -56(%rbp)
	jmp	.L69
.L102:
	movq	$34, -56(%rbp)
	jmp	.L69
.L18:
	movl	-172(%rbp), %edx
	movl	-164(%rbp), %eax
	cmpl	%eax, %edx
	jge	.L104
	movq	$69, -56(%rbp)
	jmp	.L69
.L104:
	movq	$71, -56(%rbp)
	jmp	.L69
.L21:
	movq	$21, -56(%rbp)
	jmp	.L69
.L42:
	movl	-160(%rbp), %eax
	movl	%eax, -92(%rbp)
	movq	$62, -56(%rbp)
	jmp	.L69
.L30:
	movl	-128(%rbp), %eax
	movl	%eax, -136(%rbp)
	movq	$61, -56(%rbp)
	jmp	.L69
.L10:
	movl	-164(%rbp), %eax
	movl	%eax, -108(%rbp)
	movq	$44, -56(%rbp)
	jmp	.L69
.L51:
	movl	-176(%rbp), %eax
	cmpl	%eax, -124(%rbp)
	jge	.L106
	movq	$14, -56(%rbp)
	jmp	.L69
.L106:
	movq	$27, -56(%rbp)
	jmp	.L69
.L26:
	addl	$1, -156(%rbp)
	movq	$25, -56(%rbp)
	jmp	.L69
.L14:
	movl	-172(%rbp), %eax
	movl	%eax, -140(%rbp)
	movq	$13, -56(%rbp)
	jmp	.L69
.L31:
	movl	-144(%rbp), %eax
	movl	%eax, -152(%rbp)
	movq	$19, -56(%rbp)
	jmp	.L69
.L7:
	movl	-172(%rbp), %eax
	movl	%eax, -148(%rbp)
	movq	$55, -56(%rbp)
	jmp	.L69
.L34:
	movl	-180(%rbp), %eax
	cmpl	%eax, -108(%rbp)
	jle	.L108
	movq	$42, -56(%rbp)
	jmp	.L69
.L108:
	movq	$24, -56(%rbp)
	jmp	.L69
.L64:
	movl	-172(%rbp), %eax
	movl	%eax, -116(%rbp)
	movq	$54, -56(%rbp)
	jmp	.L69
.L9:
	movl	-168(%rbp), %eax
	movl	%eax, -92(%rbp)
	movq	$62, -56(%rbp)
	jmp	.L69
.L43:
	movl	-168(%rbp), %eax
	movl	%eax, -100(%rbp)
	movq	$12, -56(%rbp)
	jmp	.L69
.L40:
	movl	-160(%rbp), %eax
	movl	%eax, -124(%rbp)
	movq	$22, -56(%rbp)
	jmp	.L69
.L15:
	movl	-104(%rbp), %eax
	movl	%eax, -84(%rbp)
	movq	$43, -56(%rbp)
	jmp	.L69
.L36:
	movl	-180(%rbp), %eax
	movl	%eax, -112(%rbp)
	movq	$59, -56(%rbp)
	jmp	.L69
.L68:
	movl	-164(%rbp), %eax
	movl	%eax, -148(%rbp)
	movq	$55, -56(%rbp)
	jmp	.L69
.L32:
	movl	-168(%rbp), %edx
	movl	-160(%rbp), %eax
	cmpl	%eax, %edx
	jge	.L110
	movq	$33, -56(%rbp)
	jmp	.L69
.L110:
	movq	$35, -56(%rbp)
	jmp	.L69
.L38:
	addl	$1, -84(%rbp)
	movq	$43, -56(%rbp)
	jmp	.L69
.L63:
	movl	-96(%rbp), %eax
	movl	%eax, -104(%rbp)
	movl	-120(%rbp), %eax
	movl	%eax, -88(%rbp)
	movq	$45, -56(%rbp)
	jmp	.L69
.L41:
	movl	-160(%rbp), %eax
	movl	%eax, -100(%rbp)
	movq	$12, -56(%rbp)
	jmp	.L69
.L46:
	cmpl	$3, -80(%rbp)
	jne	.L112
	movq	$53, -56(%rbp)
	jmp	.L69
.L112:
	movq	$25, -56(%rbp)
	jmp	.L69
.L35:
	movl	-84(%rbp), %eax
	cmpl	-136(%rbp), %eax
	jg	.L114
	movq	$56, -56(%rbp)
	jmp	.L69
.L114:
	movq	$2, -56(%rbp)
	jmp	.L69
.L67:
	addl	$1, -88(%rbp)
	movq	$45, -56(%rbp)
	jmp	.L69
.L119:
	nop
.L69:
	jmp	.L116
.L118:
	call	__stack_chk_fail@PLT
.L117:
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
