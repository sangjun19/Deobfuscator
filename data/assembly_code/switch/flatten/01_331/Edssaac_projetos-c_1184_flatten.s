	.file	"Edssaac_projetos-c_1184_flatten.c"
	.text
	.globl	_TIG_IZ_yTpW_argv
	.bss
	.align 8
	.type	_TIG_IZ_yTpW_argv, @object
	.size	_TIG_IZ_yTpW_argv, 8
_TIG_IZ_yTpW_argv:
	.zero	8
	.globl	_TIG_IZ_yTpW_argc
	.align 4
	.type	_TIG_IZ_yTpW_argc, @object
	.size	_TIG_IZ_yTpW_argc, 4
_TIG_IZ_yTpW_argc:
	.zero	4
	.globl	_TIG_IZ_yTpW_envp
	.align 8
	.type	_TIG_IZ_yTpW_envp, @object
	.size	_TIG_IZ_yTpW_envp, 8
_TIG_IZ_yTpW_envp:
	.zero	8
	.section	.rodata
.LC1:
	.string	"%.1f\n"
.LC2:
	.string	"%f"
.LC4:
	.string	" %c"
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
	subq	$656, %rsp
	movl	%edi, -628(%rbp)
	movq	%rsi, -640(%rbp)
	movq	%rdx, -648(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_yTpW_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_yTpW_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_yTpW_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 106 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-yTpW--0
# 0 "" 2
#NO_APP
	movl	-628(%rbp), %eax
	movl	%eax, _TIG_IZ_yTpW_argc(%rip)
	movq	-640(%rbp), %rax
	movq	%rax, _TIG_IZ_yTpW_argv(%rip)
	movq	-648(%rbp), %rax
	movq	%rax, _TIG_IZ_yTpW_envp(%rip)
	nop
	movq	$8, -600(%rbp)
.L40:
	cmpq	$31, -600(%rbp)
	ja	.L44
	movq	-600(%rbp), %rax
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
	.long	.L25-.L8
	.long	.L44-.L8
	.long	.L24-.L8
	.long	.L44-.L8
	.long	.L23-.L8
	.long	.L22-.L8
	.long	.L44-.L8
	.long	.L44-.L8
	.long	.L21-.L8
	.long	.L44-.L8
	.long	.L44-.L8
	.long	.L44-.L8
	.long	.L20-.L8
	.long	.L19-.L8
	.long	.L18-.L8
	.long	.L17-.L8
	.long	.L16-.L8
	.long	.L44-.L8
	.long	.L15-.L8
	.long	.L44-.L8
	.long	.L14-.L8
	.long	.L44-.L8
	.long	.L44-.L8
	.long	.L13-.L8
	.long	.L44-.L8
	.long	.L44-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L44-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L15:
	movss	-620(%rbp), %xmm0
	movss	.LC0(%rip), %xmm1
	divss	%xmm1, %xmm0
	movss	%xmm0, -604(%rbp)
	pxor	%xmm2, %xmm2
	cvtss2sd	-604(%rbp), %xmm2
	movq	%xmm2, %rax
	movq	%rax, %xmm0
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movq	$15, -600(%rbp)
	jmp	.L26
.L23:
	movl	$1, -608(%rbp)
	movl	$1, -616(%rbp)
	movq	$0, -600(%rbp)
	jmp	.L26
.L9:
	leaq	-592(%rbp), %rcx
	movl	-612(%rbp), %eax
	movslq	%eax, %rsi
	movl	-616(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	addq	%rsi, %rax
	salq	$2, %rax
	addq	%rcx, %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	addl	$1, -612(%rbp)
	movq	$31, -600(%rbp)
	jmp	.L26
.L18:
	movl	$0, -612(%rbp)
	movq	$31, -600(%rbp)
	jmp	.L26
.L17:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L41
	jmp	.L42
.L7:
	cmpl	$11, -612(%rbp)
	jg	.L28
	movq	$30, -600(%rbp)
	jmp	.L26
.L28:
	movq	$23, -600(%rbp)
	jmp	.L26
.L20:
	movl	-612(%rbp), %eax
	cmpl	-608(%rbp), %eax
	jge	.L30
	movq	$2, -600(%rbp)
	jmp	.L26
.L30:
	movq	$26, -600(%rbp)
	jmp	.L26
.L21:
	movq	$5, -600(%rbp)
	jmp	.L26
.L13:
	addl	$1, -616(%rbp)
	movq	$20, -600(%rbp)
	jmp	.L26
.L16:
	pxor	%xmm3, %xmm3
	cvtss2sd	-620(%rbp), %xmm3
	movq	%xmm3, %rax
	movq	%rax, %xmm0
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movq	$15, -600(%rbp)
	jmp	.L26
.L12:
	addl	$1, -608(%rbp)
	addl	$1, -616(%rbp)
	movq	$0, -600(%rbp)
	jmp	.L26
.L19:
	movzbl	-621(%rbp), %eax
	movsbl	%al, %eax
	cmpl	$77, %eax
	je	.L32
	cmpl	$83, %eax
	je	.L33
	jmp	.L43
.L32:
	movq	$18, -600(%rbp)
	jmp	.L35
.L33:
	movq	$16, -600(%rbp)
	jmp	.L35
.L43:
	movq	$29, -600(%rbp)
	nop
.L35:
	jmp	.L26
.L11:
	movl	$0, -612(%rbp)
	movq	$12, -600(%rbp)
	jmp	.L26
.L22:
	pxor	%xmm0, %xmm0
	movss	%xmm0, -604(%rbp)
	movss	-604(%rbp), %xmm0
	movss	%xmm0, -620(%rbp)
	leaq	-621(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	$0, -616(%rbp)
	movq	$20, -600(%rbp)
	jmp	.L26
.L25:
	cmpl	$11, -616(%rbp)
	jg	.L36
	movq	$27, -600(%rbp)
	jmp	.L26
.L36:
	movq	$13, -600(%rbp)
	jmp	.L26
.L10:
	movq	$15, -600(%rbp)
	jmp	.L26
.L24:
	movl	-612(%rbp), %eax
	movslq	%eax, %rcx
	movl	-616(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	addq	%rcx, %rax
	movss	-592(%rbp,%rax,4), %xmm0
	movss	-620(%rbp), %xmm1
	addss	%xmm1, %xmm0
	movss	%xmm0, -620(%rbp)
	addl	$1, -612(%rbp)
	movq	$12, -600(%rbp)
	jmp	.L26
.L14:
	cmpl	$11, -616(%rbp)
	jg	.L38
	movq	$14, -600(%rbp)
	jmp	.L26
.L38:
	movq	$4, -600(%rbp)
	jmp	.L26
.L44:
	nop
.L26:
	jmp	.L40
.L42:
	call	__stack_chk_fail@PLT
.L41:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	main, .-main
	.section	.rodata
	.align 4
.LC0:
	.long	1115947008
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
